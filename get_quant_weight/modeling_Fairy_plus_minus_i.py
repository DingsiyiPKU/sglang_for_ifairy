"""
ComplexNet model with Dummy Complex Semantic
backpropogation with simple autograd
"""

from typing import Optional, Tuple, Callable, Any, Dict, List
import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.generation.utils import GenerationMixin
except ImportError:
    from transformers.modeling_utils import GenerationMixin

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

try:
    from .configuration_Fairy_plus_minus_i import ComplexNetConfig
except ImportError:
    from configuration_Fairy_plus_minus_i import ComplexNetConfig


from transformers.cache_utils import Cache

import torch
from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg


logger = logging.get_logger(__name__)


class ComplexDynamicCache(Cache):
    """
    A complex cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = ComplexDynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        ComplexDynamicCache()
        ```
    """

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )
        self.key_real_cache: List[torch.Tensor] = []
        self.key_imag_cache: List[torch.Tensor] = []
        self.value_real_cache: List[torch.Tensor] = []
        self.value_imag_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.key_real_cache[layer_idx],
                self.key_imag_cache[layer_idx],
                self.value_real_cache[layer_idx],
                self.value_imag_cache[layer_idx],
            )
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (
                self.key_real_cache[layer_idx],
                self.key_imag_cache[layer_idx],
                self.value_real_cache[layer_idx],
                self.value_imag_cache[layer_idx],
            )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_real_cache)

    def update(
        self,
        key_real_states: torch.Tensor,
        key_imag_states: torch.Tensor,
        value_real_states: torch.Tensor,
        value_imag_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `ComplexDynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_real_states.shape[-2]

        # Update the cache
        if key_real_states is not None:
            if len(self.key_real_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_real_cache), layer_idx):
                    self.key_real_cache.append([])
                    self.key_imag_cache.append([])
                    self.value_real_cache.append([])
                    self.value_imag_cache.append([])
                self.key_real_cache.append(key_real_states)
                self.key_imag_cache.append(key_imag_states)
                self.value_real_cache.append(value_real_states)
                self.value_imag_cache.append(value_imag_states)
            elif (
                len(self.key_real_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_real_cache[layer_idx] = key_real_states
                self.key_imag_cache[layer_idx] = key_imag_states
                self.value_real_cache[layer_idx] = value_real_states
                self.value_imag_cache[layer_idx] = value_imag_states

            else:
                self.key_real_cache[layer_idx] = torch.cat(
                    [self.key_real_cache[layer_idx], key_real_states], dim=-2
                )
                self.key_imag_cache[layer_idx] = torch.cat([self.key_imag_cache[layer_idx], key_imag_states], dim=-2)
                self.value_real_cache[layer_idx] = torch.cat(
                    [self.value_real_cache[layer_idx], value_real_states], dim=-2
                )
                self.value_imag_cache[layer_idx] = torch.cat(
                    [self.value_imag_cache[layer_idx], value_imag_states], dim=-2
                )

        return (
            self.key_real_cache[layer_idx],
            self.key_imag_cache[layer_idx],
            self.value_real_cache[layer_idx],
            self.value_imag_cache[layer_idx],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_real_cache) == 0  # no cache in any layer
            or len(self.key_real_cache)
            <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_real_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = (
            self.key_real_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        )
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(
        self,
    ) -> Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
    ]:
        """Converts the `ComplexDynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += (
                (
                    self.key_real_cache[layer_idx],
                    self.key_imag_cache[layer_idx],
                    self.value_real_cache[layer_idx],
                    self.value_imag_cache[layer_idx],
                ),
            )
        return legacy_cache

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        num_hidden_layers: int = None,
    ) -> "ComplexDynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `ComplexDynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                (
                    key_real_states,
                    key_imag_states,
                    value_real_states,
                    value_imag_states,
                ) = past_key_values[layer_idx]
                cache.update(
                    key_real_states,
                    key_imag_states,
                    value_real_states,
                    value_imag_states,
                    layer_idx,
                )
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search.
        """
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_real_cache)):
            if self.key_real_cache[idx] != []:
                self.key_real_cache[idx] = self.key_real_cache[idx][..., :max_length, :]
                self.key_imag_cache[idx] = self.key_imag_cache[idx][..., :max_length, :]
                self.value_real_cache[idx] = self.value_real_cache[idx][
                    ..., :max_length, :
                ]
                self.value_imag_cache[idx] = self.value_imag_cache[idx][
                    ..., :max_length, :
                ]

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["ComplexDynamicCache"]:
        """Split the current instance into a list of `ComplexDynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = ComplexDynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_real_cache = [
                tensor[i : i + split_size] for tensor in self.key_real_cache
            ]
            current_split.key_imag_cache = [
                tensor[i : i + split_size] for tensor in self.key_imag_cache
            ]
            current_split.value_real_cache = [
                tensor[i : i + split_size] for tensor in self.value_real_cache
            ]
            current_split.value_imag_cache = [
                tensor[i : i + split_size] for tensor in self.value_imag_cache
            ]

            out.append(current_split)
        return out

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_batch_splits(
        cls, splits: List["ComplexDynamicCache"], num_hidden_layers: int = None
    ) -> "ComplexDynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_real_cache = [
                current.key_real_cache[idx]
                for current in splits
                if current.key_real_cache[idx] != []
            ]
            key_imag_cache = [
                current.key_imag_cache[idx]
                for current in splits
                if current.key_imag_cache[idx] != []
            ]
            value_real_cache = [
                current.value_real_cache[idx]
                for current in splits
                if current.value_real_cache[idx] != []
            ]
            value_imag_cache = [
                current.value_imag_cache[idx]
                for current in splits
                if current.value_imag_cache[idx] != []
            ]
            
            if key_real_cache != []:
                layer_keys_real = torch.cat(key_real_cache, dim=0)
                layer_keys_imag = torch.cat(key_imag_cache, dim=0)
                layer_values_real = torch.cat(value_real_cache, dim=0)
                layer_values_imag = torch.cat(value_imag_cache, dim=0)
         
                cache.update(
                    layer_keys_real,
                    layer_keys_imag,
                    layer_values_real,
                    layer_values_imag,
                    idx,
                )
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_real_cache[layer_idx] = self.key_real_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
            self.key_imag_cache[layer_idx] = self.key_imag_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
            self.value_real_cache[layer_idx] = self.value_real_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
            self.value_imag_cache[layer_idx] = self.value_imag_cache[
                layer_idx
            ].repeat_interleave(repeats, dim=0)
         
    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_real_cache[layer_idx] = self.key_real_cache[layer_idx][
                indices, ...
            ]
            self.key_imag_cache[layer_idx] = self.key_imag_cache[layer_idx][
                indices, ...
            ]
            self.value_real_cache[layer_idx] = self.value_real_cache[layer_idx][
                indices, ...
            ]
            self.value_imag_cache[layer_idx] = self.value_imag_cache[layer_idx][
                indices, ...
            ]


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

_CONFIG_FOR_DOC = "ComplexNetConfig"


class DirectionQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        phase = torch.angle(w_real + 1j * w_imag)
        real_pos = (phase >= -torch.pi / 4) & (phase < torch.pi / 4)
        real_neg = (phase >= 3 * torch.pi / 4) | (phase < -3 * torch.pi / 4)
        imag_pos = (phase >= torch.pi / 4) & (phase < 3 * torch.pi / 4)
        imag_neg = (phase >= -3 * torch.pi / 4) & (phase < -torch.pi / 4)
        real_scale = 1.0 / torch.clamp(w_real[real_pos|real_neg].abs().mean(), min=1e-5)
        imag_scale = 1.0 / torch.clamp(w_imag[imag_pos|imag_neg].abs().mean(), min=1e-5)
        
        qw_real = torch.zeros_like(w_real,dtype=w_real.dtype)
        qw_imag = torch.zeros_like(w_imag,dtype=w_imag.dtype)

        qw_real[real_pos] = 1.0
        qw_imag[imag_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_neg] = -1.0

        qw_real = qw_real / real_scale
        qw_imag = qw_imag / imag_scale
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag


def weight_quant_qat(w_real: torch.Tensor, w_imag: torch.Tensor):
    return DirectionQuantSTE.apply(w_real, w_imag)


class ComplexWeightQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w_real, w_imag):
        return weight_quant_qat(w_real, w_imag)


class ActivationQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_real: torch.Tensor, x_imag: torch.Tensor):
        real_scale = 127.0 / x_real.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=1e-5
        )
        imag_scale = 127.0 / x_imag.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=1e-5
        )

        qx_real = x_real * real_scale
        qx_real = qx_real.contiguous()
        qx_real.round_()
        qx_real.clamp_(-128, 127)
        qx_real.div_(real_scale)

        qx_imag = x_imag * imag_scale
        qx_imag = qx_imag.contiguous()
        qx_imag.round_()
        qx_imag.clamp_(-128, 127)
        qx_imag.div_(imag_scale)

        return qx_real, qx_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        # STE
        return grad_real, grad_imag


def activation_quant_qat(x_real: torch.Tensor, x_imag: torch.Tensor):
    return ActivationQuantSTE.apply(x_real, x_imag)


class ComplexActivationQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_real, x_imag):
        return activation_quant_qat(x_real, x_imag)


class HalfComplexLinear(nn.Module):
    """
    HalfComplexLinear is a linear layer that only outputs real_output.
    """

    def __init__(self, in_features: int, out_features: int):
        super(HalfComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_real = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.act_quantizer = ComplexActivationQuantizer()
        self.weight_quantizer = ComplexWeightQuantizer()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        qw_real, qw_imag = self.weight_quantizer(self.weight_real, self.weight_imag)
        qx_real, qx_imag = self.act_quantizer(x_real, x_imag)

        out_real = F.linear(qx_real, qw_real) + F.linear(qx_imag, qw_imag)
      
        return out_real


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.weight_real = nn.Parameter(
        #     torch.empty(self.out_features, self.in_features)
        # )
        # self.weight_imag = nn.Parameter(
        #     torch.empty(self.out_features, self.in_features)
        # )
        self.weight_real = nn.Parameter(
            torch.empty(self.out_features, self.in_features,dtype= torch.bfloat16)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(self.out_features, self.in_features,dtype=torch.bfloat16)
        )
        self.act_quantizer = ComplexActivationQuantizer()
        self.weight_quantizer = ComplexWeightQuantizer()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor,if_print = False) -> torch.Tensor:

        qw_real, qw_imag = self.weight_quantizer(self.weight_real, self.weight_imag)
        qx_real, qx_imag = self.act_quantizer(x_real, x_imag)
        with torch.no_grad():
        # .copy_() 是一个 in-place 操作，将 qw 的数据拷贝到 self.weight 的内存中
            self.weight_real.copy_(qw_real)
            self.weight_imag.copy_(qw_imag)        

        
        '''checkpoint = {
    'inputs': {'real': qx_real, 'imag': qx_imag},
    'weights':{'real':qw_real,'imag':qw_imag}
}
        torch.save(checkpoint,'/home/wangyuxing/complex_layer_snapshot_input.pt')
        print("量化后参数保存完毕")'''
        out_real = F.linear(qx_real, qw_real) + F.linear(qx_imag, qw_imag)
        out_imag = F.linear(qx_real, qw_imag) - F.linear(qx_imag, qw_real)

        return out_real, out_imag


class ComplexNetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight_real = nn.Parameter(torch.ones(hidden_size))
        self.weight_imag = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states_real: torch.Tensor, hidden_states_imag: torch.Tensor
    ):
        input_dtype = hidden_states_real.dtype

        # do the RMS computations in float32 for numerical stability
        hidden_states_real = hidden_states_real.to(torch.float32)
        hidden_states_imag = hidden_states_imag.to(torch.float32)
        magnitude = torch.mean(
            hidden_states_real**2 + hidden_states_imag**2, dim=-1, keepdim=True
        )
        variance = torch.rsqrt(magnitude + self.variance_epsilon)

        hidden_states_real = hidden_states_real * variance
        hidden_states_imag = hidden_states_imag * variance

        # cast weights to the computation dtype to avoid unexpected upcasts
        weight_real = self.weight_real.to(hidden_states_real.dtype)
        weight_imag = self.weight_imag.to(hidden_states_imag.dtype)

        rmsnorm_out_real = weight_real * hidden_states_real
        rmsnorm_out_imag = weight_imag * hidden_states_imag


        return rmsnorm_out_real.to(input_dtype), rmsnorm_out_imag.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(ComplexNetRMSNorm)


class ComplexNetMLP(nn.Module):
    def __init__(self, config: ComplexNetConfig,layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.im_size = self.config.intermediate_size

        self.gate_proj = ComplexLinear(self.hidden_size, self.im_size)
        self.up_proj = ComplexLinear(self.hidden_size, self.im_size)
        self.down_proj = ComplexLinear(self.im_size, self.hidden_size)

        self.ffn_layernorm = ComplexNetRMSNorm(self.im_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def complex_relu2(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        mask = torch.logical_and(x_real < 0, x_imag < 0)
        x_real[mask] = 0
        x_imag[mask] = 0
        x_real = x_real**2
        x_imag = x_imag**2
        return x_real, x_imag

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        gate_proj_real, gate_proj_imag = self.gate_proj(x_real, x_imag)
        
        
        activated_real, activated_imag = self.complex_relu2(
            gate_proj_real, gate_proj_imag
        )
        
        up_proj_real, up_proj_imag = self.up_proj(x_real, x_imag)
        
        up_proj_activated_real = (
            activated_real * up_proj_real + activated_imag * up_proj_imag
        )
        up_proj_activated_imag = (
            activated_real * up_proj_imag - activated_imag * up_proj_real
        )
        ln_real, ln_imag = self.ffn_layernorm(
            up_proj_activated_real, up_proj_activated_imag
        )

        out_real, out_imag = self.down_proj(ln_real, ln_imag)
        
        return out_real, out_imag


class ComplexNetRotaryEmbedding(nn.Module):
    def __init__(self, config: ComplexNetConfig):
        super().__init__()

        self.config = config
        self.base = self.config.rope_theta
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.max_seq_len_cached = self.config.max_position_embeddings

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self):
        base = self.base
        head_dim = self.hidden_size // self.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, dtype=torch.int64) / head_dim)
        )
        return inv_freq

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor, hidden_states_type: torch.dtype
    ) -> tuple:
        batch_size = position_ids.shape[0]
        position_ids = position_ids[:, None, :].to(torch.float32)
        if self.inv_freq.dim() == 1:
            self.inv_freq = (
                self.inv_freq[None, :, None]
                .expand(batch_size, -1, 1)
                .to(position_ids.device)
            )

        if position_ids.shape[0] > self.max_seq_len_cached:
            print(f"Truncate position_ids within max_seq_len_cached.")
            position_ids = position_ids[: self.max_seq_len_cached]
        theta = (self.inv_freq.to(position_ids.dtype) @ position_ids).transpose(1, 2)
        cos_emb = torch.cos(theta).to(hidden_states_type)
        sin_emb = torch.sin(theta).to(hidden_states_type)

        return cos_emb, sin_emb


def _apply_rotary_pos_emb(
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    k_real: torch.Tensor,
    k_imag: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
) -> tuple:

    def _apply_rotation(
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        cos_emb: torch.Tensor,
        sin_emb: torch.Tensor,
    ) -> torch.Tensor:
        cos_emb = cos_emb.unsqueeze(1)
        sin_emb = sin_emb.unsqueeze(1)

        rotated_x_real = x_real * cos_emb - x_imag * sin_emb
        rotated_x_imag = x_real * sin_emb + x_imag * cos_emb

        return rotated_x_real, rotated_x_imag

    rotated_q_real, rotated_q_imag = _apply_rotation(q_real, q_imag, cos_emb, sin_emb)
    rotated_k_real, rotated_k_imag = _apply_rotation(k_real, k_imag, cos_emb, sin_emb)

    return rotated_q_real, rotated_q_imag, rotated_k_real, rotated_k_imag


def repeat_kv(
    hidden_states_real: torch.Tensor,
    hidden_states_imag: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_length, head_dim = hidden_states_real.shape

    if num_key_value_groups == 1:
        return hidden_states_real, hidden_states_imag

    hidden_states_real = hidden_states_real[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    )
    hidden_states_imag = hidden_states_imag[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    )

    hidden_states_real = hidden_states_real.reshape(
        batch_size, num_key_value_heads * num_key_value_groups, seq_length, head_dim
    )
    hidden_states_imag = hidden_states_imag.reshape(
        batch_size, num_key_value_heads * num_key_value_groups, seq_length, head_dim
    )
    return hidden_states_real, hidden_states_imag


def repeat_kv_for_real(
    hidden_states_real: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_length, head_dim = hidden_states_real.shape

    if num_key_value_groups == 1:
        return hidden_states_real

    hidden_states_real = hidden_states_real[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    )

    hidden_states_real = hidden_states_real.reshape(
        batch_size, num_key_value_heads * num_key_value_groups, seq_length, head_dim
    )
    return hidden_states_real


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_only_for_real(
    q_real: torch.Tensor,
    k_real: torch.Tensor,
    cos_emb: torch.Tensor,
    sin_emb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cos_emb = cos_emb.unsqueeze(1)
    sin_emb = sin_emb.unsqueeze(1)
    q_embed = (q_real * cos_emb) + (_rotate_half(q_real) * sin_emb)
    k_embed = (k_real * cos_emb) + (_rotate_half(k_real) * sin_emb)
    return q_embed, k_embed


class ComplexNetAttentionBase(nn.Module):
    def __init__(self, config: ComplexNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_dropout = self.config.attention_dropout

        self.hidden_size = self.config.hidden_size
        self.num_attn_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attn_heads

        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attn_heads // self.config.num_key_value_heads
        )

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.scaling = self.head_dim**-0.5

        self.is_causal = True
        self.rms_norm_eps = self.config.rms_norm_eps

        self.q_proj = ComplexLinear(
            self.hidden_size, self.num_attn_heads * self.head_dim
        )
        self.k_proj = ComplexLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim
        )
        self.v_proj = ComplexLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim
        )
        self.o_proj = ComplexLinear(self.hidden_size, self.hidden_size)

        self.rotary_emb = ComplexNetRotaryEmbedding(self.config)
        self.attn_layernorm = ComplexNetRMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    def forward(
        self,
        hidden_states_real: torch.Tensor,
        hidden_states_imag: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_mask_real: Optional[torch.Tensor] = None,
        attn_mask_imag: Optional[torch.Tensor] = None,
        past_key_value: Optional[ComplexDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states_real.shape[:-1]
        q_shape = (*input_shape, self.num_attn_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        q_real = self.q_proj(hidden_states_real, hidden_states_imag)
        q_real = q_real.view(q_shape).transpose(1, 2)

        k_real = self.k_proj(hidden_states_real, hidden_states_imag)
        k_imag = None  
        k_real = k_real.view(kv_shape).transpose(1, 2)

        v_real, v_imag = self.v_proj(hidden_states_real, hidden_states_imag)
        v_real = v_real.view(kv_shape).transpose(1, 2)
        v_imag = v_imag.view(kv_shape).transpose(1, 2)

        cos_emb, sin_emb = position_embeddings
        q_real, k_real = _apply_rotary_pos_emb_only_for_real(
            q_real, k_real, cos_emb, sin_emb
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_emb,
                "cos": cos_emb,
                "cache_position": cache_position,
            }
            k_real, k_imag, v_real, v_imag = past_key_value.update(
                k_real, k_imag, v_real, v_imag, self.layer_idx, cache_kwargs
            )

        k_real = repeat_kv_for_real(k_real, self.num_key_value_groups)
        v_real, v_imag = repeat_kv(v_real, v_imag, self.num_key_value_groups)

        attn_weights_real = (q_real @ k_real.transpose(2, 3)) * self.scaling
        if attn_mask_real is not None:
            causal_mask_real = attn_mask_real[:, :, :, : k_real.shape[-2]]
            attn_weights_real = attn_weights_real + causal_mask_real

        attn_weights = attn_weights_real
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q_real.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attn_dropout, training=self.training
        )
        attn_output_real = (
            torch.matmul(attn_weights, v_real)
            .transpose(1, 2)
            .contiguous()
            .reshape(input_shape[0], input_shape[1], self.hidden_size)
        )
        attn_output_imag = (
            torch.matmul(attn_weights, v_imag)
            .transpose(1, 2)
            .contiguous()
            .reshape(input_shape[0], input_shape[1], self.hidden_size)
        )

        attn_output_real, attn_output_imag = self.attn_layernorm(
            attn_output_real, attn_output_imag
        )
        attn_output_real, attn_output_imag = self.o_proj(
            attn_output_real, attn_output_imag
        )

        return (
            attn_output_real,
            attn_output_imag,
            attn_weights_real,
            None,  # attn_weights_imag
            past_key_value,
        )


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    module,
    query_layer,
    key_layer,
    value_layer,
    attention_mask,
    query_length,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )

    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(
                batch_size * kv_seq_len, module.num_attn_heads, head_dim
            ),
            indices_k,
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, attention_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def eager_attention_forward(
    module: nn.Module,
    q_cat: torch.Tensor,
    k_cat: torch.Tensor,
    v_cat: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # for_real func only handle one input
    k_cat = repeat_kv_for_real(k_cat, module.num_key_value_groups)
    v_cat = repeat_kv_for_real(v_cat, module.num_key_value_groups)

    attn_weights_real = (q_cat @ k_cat.transpose(2, 3)) * scaling
    if attn_mask is not None:
        causal_mask_real = attn_mask[:, :, :, : k_cat.shape[-2]]
        attn_weights_real = attn_weights_real + causal_mask_real

        attn_weights = attn_weights_real
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q_cat.dtype
        )
      
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, v_cat).transpose(1, 2).contiguous()

    return attn_output, None, None


def flash_attention_forward(
    module: nn.Module,
    q_cat: torch.Tensor,
    k_cat: torch.Tensor,
    v_cat: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    **kwargs,
):
    def transpose_hidden_states(*hidden_states: torch.Tensor):
        return [tensor.transpose(1, 2) for tensor in hidden_states]

    (q_cat, k_cat, v_cat) = transpose_hidden_states(q_cat, k_cat, v_cat)
    query_len = 1
    query_len = q_cat.shape[1]
    input_dtype = q_cat.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = module.q_proj.weight_real.dtype

        def dtype_cast(*tensors: torch.Tensor):
            return [tensor.to(target_dtype) for tensor in tensors]

        (q_cat, k_cat, v_cat) = dtype_cast(q_cat, k_cat, v_cat)
    if not module._flash_attn_uses_top_left_mask:
        causal = module.is_causal
    else:
        causal = module.is_causal and query_len != 1

    if attn_mask is not None:
        batch_size = q_cat.shape[0]
        (
            q_cat,
            k_cat,
            v_cat,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = _upad_input(
            module,
            q_cat,
            k_cat,
            v_cat,
            attn_mask,
            query_len,
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        def mask_complex_flash_attn(q, k, v):
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )


        attn_output_unpad = mask_complex_flash_attn(q_cat, k_cat, v_cat)
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_len)
    else:

        def unmask_complex_flash_attn(q, k, v):
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        attn_output = unmask_complex_flash_attn(q_cat, k_cat, v_cat)
    return attn_output, None, None


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "flash_attention_2": flash_attention_forward,
}


class ComplexNetAttention(ComplexNetAttentionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states_real: torch.Tensor,
        hidden_states_imag: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_mask_real: Optional[torch.Tensor] = None,
        attn_mask_imag: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states_real.shape[:-1]
        q_shape = (*input_shape, self.num_attn_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        def transpose_hidden_states(*hidden_states: torch.Tensor):
            return [tensor.transpose(1, 2) for tensor in hidden_states]

        q_real, q_imag = self.q_proj(hidden_states_real, hidden_states_imag,if_print = (self.layer_idx == 0))
        k_real, k_imag = self.k_proj(hidden_states_real, hidden_states_imag)
    
        v_real, v_imag = self.v_proj(hidden_states_real, hidden_states_imag)
        (q_real, q_imag, k_real, k_imag, v_real, v_imag) = transpose_hidden_states(
            q_real.view(q_shape),
            q_imag.view(q_shape),
            k_real.view(kv_shape),
            k_imag.view(kv_shape),
            v_real.view(kv_shape),
            v_imag.view(kv_shape),
        )

        cos_emb, sin_emb = position_embeddings
 
        q_real, q_imag, k_real, k_imag = _apply_rotary_pos_emb(
            q_real, q_imag, k_real, k_imag, cos_emb, sin_emb
        )       
        '''print(q_real.shape)'''
        '''checkpoint = {
    'k': {'real': k_real, 'imag': k_imag},
    'q':{'real':q_real,'imag':q_imag},
}
        torch.save(checkpoint,'/home/wangyuxing/q.pt')
        print("量化后参数保存完毕")
        import sys
        sys.exit()'''


        
        
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_emb,
                "cos": cos_emb,
                "cache_position": cache_position,
            }
            k_real, k_imag, v_real, v_imag = past_key_value.update(
                k_real, k_imag, v_real, v_imag, self.layer_idx, cache_kwargs
            )
        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                raise ValueError(
                    f"Unsupported attention implementation: {self.config._attn_implementation}. Supported implementations are: {list(ALL_ATTENTION_FUNCTIONS.keys())}."
                )
            elif self.config._attn_implementation == "flash_attention_2":
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
            else:
                raise ValueError(
                    f"Unsupported attention implementation: {self.config._attn_implementation}. Supported implementations are: {list(ALL_ATTENTION_FUNCTIONS.keys())}."
                )
        cat_q = torch.cat([q_real, q_imag], dim=-1).reshape(
            input_shape[0], self.num_attn_heads, input_shape[1], 2 * self.head_dim
        )
        B, H, S, D = k_real.shape
        cat_k = torch.cat([k_real, k_imag], dim=-1).reshape(
            B, H, S, 2 * D
        )
        cat_v = torch.cat([v_real, v_imag], dim=-1).reshape(
            B, H, S, 2 * D
        )
        attn_output, attn_weights_real, attn_weights_imag = attention_interface(
            self,
            cat_q,
            cat_k,
            cat_v,
            attn_mask_real,
            scaling=self.scaling,
            dropout=self.attn_dropout if self.training else 0.0,
            **kwargs,
        )        

        attn_output_real, attn_output_imag = torch.chunk(attn_output, 2, dim=-1)
        attn_output_real = attn_output_real.reshape(
            input_shape[0], input_shape[1], self.hidden_size
        ).contiguous()
        attn_output_imag = attn_output_imag.reshape(
            input_shape[0], input_shape[1], self.hidden_size
        ).contiguous()

        attn_output_real, attn_output_imag = self.attn_layernorm(
            attn_output_real, attn_output_imag
        )
        '''print(f"形状为{attn_output_real.shape}")
        checkpoint = {
    'attn': {'real':attn_output_real, 'imag': attn_output_imag},
}
        torch.save(checkpoint,'/home/wangyuxing/q.pt')
        print("量化后参数保存完毕")
        import sys
        sys.exit()'''

        attn_output_real, attn_output_imag = self.o_proj(
            attn_output_real, attn_output_imag
        )

        if not output_attentions:
            attn_weights_real = None
            attn_weights_imag = None

        '''checkpoint = {
    'attn': {'real':attn_output_real, 'imag': attn_output_imag},
}
        torch.save(checkpoint,'/home/wangyuxing/q.pt')
        print("量化后参数保存完毕")
        import sys
        sys.exit()'''
        return (
            attn_output_real,
            attn_output_imag,
            attn_weights_real,
            attn_weights_imag,
            past_key_value,
        )


class ComplexNetDecoderLayer(nn.Module):
    def __init__(self, config: ComplexNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size

        self.self_attn = ComplexNetAttention(config=config, layer_idx=layer_idx)
        self.mlp = ComplexNetMLP(config,layer_idx = layer_idx)
        self.pre_layernorm = ComplexNetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_layernorm = ComplexNetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx =  layer_idx
    def forward(
        self,
        hidden_states_real: torch.Tensor,
        hidden_states_imag: torch.Tensor,
        attention_mask_real: Optional[torch.Tensor] = None,
        attention_mask_imag: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual_real = hidden_states_real
        residual_imag = hidden_states_imag

        hidden_states_real, hidden_states_imag = self.pre_layernorm(
            hidden_states_real, hidden_states_imag
        )
        (
            hidden_states_real,
            hidden_states_imag,
            attn_weights_real,
            attn_weights_imag,
            present_key_value,
        ) = self.self_attn(
            hidden_states_real=hidden_states_real,
            hidden_states_imag=hidden_states_imag,
            position_embeddings=position_embeddings,
            attn_mask_real=attention_mask_real,
            attn_mask_imag=attention_mask_imag,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states_real = residual_real + hidden_states_real
        hidden_states_imag = residual_imag + hidden_states_imag

        residual_real = hidden_states_real
        residual_imag = hidden_states_imag
        


        hidden_states_real, hidden_states_imag = self.post_layernorm(
            hidden_states_real, hidden_states_imag
        )
        
        
        hidden_states_real, hidden_states_imag = self.mlp(
            hidden_states_real, hidden_states_imag
        )
        hidden_states_real = residual_real + hidden_states_real
        hidden_states_imag = residual_imag + hidden_states_imag

        outputs = (
            hidden_states_real,
            hidden_states_imag,
        )

        if output_attentions:
            outputs += (
                attn_weights_real,
                attn_weights_imag,
            )

        if use_cache:
            outputs += (present_key_value,)

        return outputs


logger = logging.get_logger(__name__)


class ComplexNetLM(PreTrainedModel, GenerationMixin):
    config_class = ComplexNetConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: ComplexNetConfig):
        super().__init__(config=config)
        self.config = config
        self.n_vocab = self.config.vocab_size
        self.max_position_embeddings = self.config.max_position_embeddings
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.use_cache = self.config.use_cache
        self.token_embeddings_real = nn.Embedding(self.n_vocab, self.hidden_size)
        self.token_embeddings_imag = nn.Embedding(self.n_vocab, self.hidden_size)
        self.final_norm = ComplexNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer = nn.ModuleList(
            [
                ComplexNetDecoderLayer(config, layer_idx)
                for layer_idx in range(self.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.rotary_emb = ComplexNetRotaryEmbedding(self.config)
        self.lm_head = nn.Linear(self.hidden_size * 2, self.n_vocab, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ComplexLinear):
            std = std / math.sqrt(2)
            torch.nn.init.normal_(module.weight_real, mean=0.0, std=std)
            torch.nn.init.normal_(module.weight_imag, mean=0.0, std=std)

    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        token_embeddings_real = self.token_embeddings_real(input_ids)
        token_embeddings_imag = self.token_embeddings_imag(input_ids)

        return token_embeddings_real, token_embeddings_imag

    def token_logits(
        self,
        x_real: torch.FloatTensor,
        x_imag: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # catenate the real and imaginary parts
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        logits = self.lm_head(x_cat)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        #print(f'input_id 为{input_ids}')
        #print(f"DEBUG: Current attention implementation: {self.config._attn_implementation}")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        x_real, x_imag = self.embed(input_ids, attention_mask)
        
        #print(f'model输入hidden_states实部{x_real}')
        #print(f'model输入hidden_states虚部{x_imag}')
        if use_cache and past_key_values is None:
            past_key_values = ComplexDynamicCache()
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + x_real.shape[1],
                device=x_real.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, x_real, cache_position, past_key_values, output_attentions
        )
        position_embeddings = self.rotary_emb(position_ids, x_real.dtype)
        all_hidden_states_real = []
        all_hidden_states_imag = []
        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(layer_module.__call__, **kwargs),
                    x_real,
                    x_imag,
                    causal_mask,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states_real=x_real,
                    hidden_states_imag=x_imag,
                    attention_mask_real=causal_mask,
                    attention_mask_imag=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            x_real, x_imag = layer_outputs[:2]

            if output_attentions:
                all_hidden_states_real.append(layer_outputs[2])
                all_hidden_states_imag.append(layer_outputs[3])

        x_real, x_imag = self.final_norm(x_real, x_imag)
        
        '''ckpt = torch.load('/home/wangyuxing/q.pt',map_location='cuda')
        out_real,out_imag = ckpt['outputs']["real"], ckpt['outputs']["imag"]
        print(out_real.shape)
       # print(f"最终model第一个token输出差异为{(out[0]-x_real[0]).abs().mean()}")
       # print(f"最终model第一个token输出差异为{torch.max((out[0]-x_real[0]).abs())}")
        print("使用eager和flash_attention的输出差异如下")
        print(f"最终model第二个token实部输出最大差异为{torch.max((out_imag-x_imag).abs())}")
        print(f"最终model第二个token实部输出差异为{(out_imag-x_imag).abs().mean()*2}")
        print(f"最终model第二个token实部输出相对误差为{(((out_imag-x_imag).abs().mean()*2)/ out_real.abs().mean())*100}%")
        print(f"最终model第二个token虚部输出最大差异为{torch.max((out_real-x_real).abs())}")
        print(f"最终model第二个token虚部输出差异为{(out_real-x_real).abs().abs().mean()*2}")
        print(f"最终model第二个token虚部输出相对误差为{(((out_imag-x_imag).abs().mean()*2)/ out_imag.abs().mean())*100}%")
        import sys
        sys.exit()'''
        
        
        
        
        
        '''print(x_real)
        checkpoint = {
    'outputs': {'real':x_real , 'imag': x_imag},
}
        torch.save(checkpoint,'/home/wangyuxing/q.pt')
        print("模型保存完毕")
        import sys
        sys.exit()'''
        
        
        logits = self.token_logits(x_real, x_imag)
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask
