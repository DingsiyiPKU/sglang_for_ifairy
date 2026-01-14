'''注意事项:
1,linear,mlp层的参数名称不匹配,需要调整,注意参数的位置
2,mlp层的tp逻辑可以优化（可选）
5,load_kv_cache_scales
6，attn层分布式rms'''
from tqdm import tqdm
import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import torch
from torch import nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch #, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.utils import make_layers
from math import sqrt

#torch.set_printoptions(edgeitems=4,sci_mode = False)


logger = logging.getLogger(__name__)
Complexconfig = None

def add_prefix(A : str, B:str) -> str:
    return A + B
def complex_relu2(x_real: torch.Tensor, x_imag: torch.Tensor) ->  Tuple[torch.Tensor,torch.Tensor]:
    # 1. 稀疏化：仅当实部和虚部同时小于0（第三象限）时，置为0
    mask = torch.logical_and(x_real < 0, x_imag < 0)
    x_real = torch.where(mask, 0.0, x_real)
    x_imag = torch.where(mask, 0.0, x_imag)
    # 2. 非线性：对所有元素进行平方
    x_real = x_real**2
    x_imag = x_imag**2
    return x_real, x_imag
class ComplexRelu2AndMul(nn.Module):
    def __init__(self):
        super().__init__()
   
    def forward(self, gate_real:torch.Tensor,gate_imag:torch.Tensor,up_real:torch.Tensor,up_imag:torch.Tensor) ->  Tuple[torch.Tensor,torch.Tensor]:
        gate_real,gate_imag = complex_relu2(gate_real,gate_imag)
       
        output_real = gate_real * up_real + gate_imag * up_imag
        output_imag = gate_real * up_imag - gate_imag * up_real
       
        return output_real,output_imag
   
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
    def init(self):
        super().init()
    def forward(self, x_real, x_imag):
        return activation_quant_qat(x_real, x_imag)
   
def IntergrateRealAndImag(real_product: torch.Tensor,imag_product : torch.Tensor ,splite_dim :int,
                          need_split:bool = True ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor] :
   
    r_r,r_i = real_product.split(splite_dim, dim=-1)
    i_r,i_i = imag_product.split(splite_dim, dim=-1)
    r_r.add_(i_i)
    r_i.sub_(i_r)
    if need_split:
        return r_r,r_i    # 返回1536 * 2
    else:
        return real_product   #返回 3072
       
from sglang.srt.layers.linear import(
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
)
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool=False ,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        "在CPU创建,GPU执行"
        c = torch.arange(0, self.rotary_dim, 2, dtype=torch.int64, device="cuda")
        inv_freq = 1.0 / (base ** (c / self.rotary_dim))
        
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        #inv_freq = inv_freq.to(dtype=torch.bfloat16)
        t = torch.arange(self.max_position_embeddings, dtype=torch.int64,device=inv_freq.device)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key   

   
  
class ComplexGetrope(nn.Module):
    def __init__(self, head_dim:int =96,max_position_embeddings:int=2048 ,rope_theta: int = 10000,basescaling: Optional[Dict[str, Any]] = None,rope_scaling: Optional[Dict[str, Any]] = None,if_print:bool = False) ->None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_emb = RotaryEmbedding(
            self.head_dim * 2,
            rotary_dim=self.head_dim * 2,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.if_print = if_print
    def getInterleaving(self, real : torch.Tensor,imag : torch.Tensor) -> torch.Tensor:
        # real,imag shape: [batch_size, seq_len, num_heads, head_dim]
        combined = torch.stack([real, imag], dim=-1)  # shape: [batch_size, seq_len, num_heads, head_dim, 2]
       
        combined=combined.contiguous()
       
        interleaved = combined.view(*real.shape[:-1], -1)  # shape: [batch_size, seq_len, num_heads, head_dim * 2]
            
        return interleaved
       
    def getDeinterleaving(self, interleaved : torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # interleaved shape: [batch_size, seq_len, num_heads, head_dim * 2]
        interleaved+interleaved.contiguous()
        reshaped = interleaved.view(*interleaved.shape[:-1], -1, 2)  # shape: [batch_size, seq_len, num_heads, head_dim, 2]
        real = reshaped[..., 0]  # shape: [batch_size, seq_len, num_heads, head_dim]
        imag = reshaped[..., 1]  # shape: [batch_size, seq_len, num_heads, head_dim]
        return real, imag    
       
   
    def forward(self,positions: torch.Tensor,q_real: torch.Tensor,q_imag:torch.Tensor,k_real: torch.Tensor,k_imag:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        q_real_and_imag = self.getInterleaving(q_real,q_imag)
        k_real_and_imag = self.getInterleaving(k_real,k_imag)
        q_rotated, k_rotated = self.rotary_emb(positions, q_real_and_imag, k_real_and_imag)
        q_rotated_real,q_rotated_imag = self.getDeinterleaving(q_rotated)
        k_rotated_real,k_rotated_imag = self.getDeinterleaving(k_rotated)
        return q_rotated_real,q_rotated_imag,k_rotated_real,k_rotated_imag
class RMSNormbase(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype  
        if residual is not None:
            x = x + residual
            residual = x 
        weight = self.weight
        
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True) * 2
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * weight 
        if residual is None:
            return x.to(orig_dtype)
        else:
            return x.to(orig_dtype), residual
class ComplexNetRMSNorm(nn.Module):
    def __init__(self, hidden_size:int ,eps:float =  1e-05,):
        super().__init__()
        self.weight = RMSNormbase(hidden_size * 2,eps)
       
    def forward(self,hidden_states_real: torch.Tensor,hidden_states_imag: torch.Tensor,residual_real : torch.Tensor = None,residual_imag : torch.Tensor = None,) :
       
        x = torch.cat([hidden_states_real, hidden_states_imag], dim=-1)
        residual = torch.cat([residual_real, residual_imag], dim=-1) if (residual_real is not None and residual_imag is not None) else None
        if residual is not None:
            normalized_x,residual = self.weight(x, residual)
            hidden_states_real,hidden_states_imag = torch.chunk(normalized_x, 2, dim=-1)
            residual_real,residual_imag = torch.chunk(residual, 2, dim=-1)
            return hidden_states_real,hidden_states_imag,residual_real,residual_imag
        else:
            normalized_x = self.weight(x)
            hidden_states_real,hidden_states_imag = torch.chunk(normalized_x, 2, dim=-1)
            return hidden_states_real,hidden_states_imag
   
   
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",)-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
   
        self.weight = ColumnParallelLinear(
                input_size = in_features ,
                output_size= out_features * 2,
                bias= False,
                quant_config=quant_config,
                prefix=prefix,
            )
        self.quantizer = ComplexActivationQuantizer()
       
    def forward(self, input_real:torch.Tensor,input_imag:torch.Tensor,) -> Tuple[torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
       
#        input_real,input_imag = self.quantizer(input_real,input_imag)
       
        input_real_and_imag  = torch.cat([input_real, input_imag], dim=0)
        Merged_output,_  = self.weight(input_real_and_imag)   #输出维度为8192   x_r [W_r,W_i]    -> x_r* w_r + x_i + w_i    
                                                              #                x_i [W_r,W_i]    -> x_i * w_r - x_r * w_i
        real_product, imag_product =  torch.chunk(Merged_output, 2, dim=0)
       
        return IntergrateRealAndImag(real_product, imag_product ,self.out_features)
    
class ComplexLinearBase(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",)-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
   
        self.weight_real = ColumnParallelLinear(
                input_size = in_features ,
                output_size= out_features ,
                bias= False,
                quant_config=quant_config,
                prefix=prefix,
            )
        
        self.weight_imag = ColumnParallelLinear(
                input_size = in_features ,
                output_size= out_features ,
                bias= False,
                quant_config=quant_config,
                prefix=prefix,
            )
        
        self.quantizer = ComplexActivationQuantizer()
       
    def forward(self, input_real:torch.Tensor,input_imag:torch.Tensor,if_print : bool = False) -> Tuple[torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"

        input_real,input_imag = self.quantizer(input_real,input_imag)
        
        r_r,_ =  self.weight_real(input_real)
        i_i,_ =  self.weight_imag(input_imag)
        x_real = r_r + i_i
        
        r_i,_ = self.weight_imag(input_real)
        i_r,_ = self.weight_real(input_imag)
        x_imag = r_i - i_r

        return x_real,x_imag   
    

class  ComplexUpLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",)-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        '''self.weight = MergedColumnParallelLinear(
                input_size = in_features,
                output_sizes=[out_features *2, out_features * 2],
                bias= False,
                quant_config=quant_config,
                gather_output=True,
                prefix=prefix,
            )'''
        self.gate_weight = ComplexLinearBase(
                in_features = in_features ,
                out_features= out_features ,
                quant_config=quant_config,
                prefix=prefix,
        )
        self.up_weight = ComplexLinearBase(
                in_features = in_features ,
                out_features= out_features ,
                quant_config=quant_config,
                prefix=prefix,
        )

       
    def forward(self, input_real:torch.Tensor,
                input_imag:torch.Tensor,if_print : bool = False) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
       
        '''input_real_quant,input_imag_quant = self.quantizer(input_real,input_imag)
        input_real_and_imag  = torch.cat([input_real_quant, input_imag_quant], dim=0)
        Merged_output,_  = self.weight(input_real_and_imag)
       
        real_product, imag_product =  torch.chunk(Merged_output, 2, dim=0)
       
        Gate_real_product,Up_real_product = real_product.split(self.out_features * 2, dim=-1)
        Gate_imag_product,Up_imag_product = imag_product.split(self.out_features * 2, dim=-1)
       
        Gate_real,Gate_imag = IntergrateRealAndImag(Gate_real_product, Gate_imag_product ,self.out_features)
        Up_real,Up_imag = IntergrateRealAndImag(Up_real_product, Up_imag_product ,self.out_features)'''

        Gate_real,Gate_imag = self.gate_weight(input_real,input_imag)

        Up_real,Up_imag = self.up_weight(input_real,input_imag)
       
        return Gate_real,Gate_imag,Up_real,Up_imag 
       
               
'''class ComplexQKVLinear(nn.Module):
    def __init__(self, head_dim:int , total_num_heads:int, total_num_kv_heads:int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",) -> None:
        super().__init__()
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = head_dim * total_num_heads
        self.q_real_imag_size = self.head_dim * self.total_num_heads  *2
       
        self.kv_real_imag_size = self.head_dim * self.total_num_kv_heads *2
       
       
       
        self.weight = QKVParallelLinear(
            self.hidden_size ,
            self.head_dim * 2,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias= False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.quantizer = ComplexActivationQuantizer()
    def forward(self, input_real: torch.Tensor,input_imag: torch.Tensor,) -> Tuple [torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"
        input_real,input_imag = self.quantizer(input_real,input_imag)
        input_real_and_imag  = torch.cat([input_real, input_imag], dim=0)
        Merged_qkv,_  = self.weight(input_real_and_imag)  
        qkv_real_product,qkv_imag_product = torch.chunk(Merged_qkv, 2, dim=0)
        q_real_product, k_real_product, v_real_product = qkv_real_product.split(
            [self.q_real_imag_size, self.kv_real_imag_size, self.kv_real_imag_size], dim=-1
        )
        q_imag_product, k_imag_product, v_imag_product = qkv_imag_product.split(
            [self.q_real_imag_size, self.kv_real_imag_size, self.kv_real_imag_size], dim=-1
        )
       
        q_real,q_imag = IntergrateRealAndImag(q_real_product,q_imag_product,self.q_real_imag_size//2)
       
        k_real,k_imag = IntergrateRealAndImag(k_real_product,k_imag_product,self.kv_real_imag_size//2)
        v_real_imag = IntergrateRealAndImag(v_real_product,v_imag_product,self.kv_real_imag_size//2,need_split=False)
       
        return q_real,q_imag,k_real,k_imag,v_real_imag'''
        
class ComplexQKVLinear(nn.Module):
    def __init__(self, head_dim:int , total_num_heads:int, total_num_kv_heads:int, quant_config: Optional[QuantizationConfig] = None, prefix: str = "",) -> None:
        super().__init__()
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = self.head_dim * self.total_num_heads
        self.q_size = self.head_dim * self.total_num_heads  
       
        self.kv_size = self.head_dim * self.total_num_kv_heads 
       
        self.q_weight = ComplexLinearBase(in_features=self.hidden_size,out_features=self.hidden_size)
        self.k_weight = ComplexLinearBase(in_features=self.hidden_size,out_features=self.kv_size)
        self.v_weight = ComplexLinearBase(in_features=self.hidden_size,out_features=self.kv_size)
    def forward(self, input_real: torch.Tensor,input_imag: torch.Tensor,) -> Tuple [torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        assert input_real.size() == input_imag.size() ,"Shape mismatch"

        q_real,q_imag = self.q_weight(input_real,input_imag)       
        k_real,k_imag = self.k_weight(input_real,input_imag)
        v_real,v_imag = self.v_weight(input_real,input_imag)
    
        return q_real,q_imag,k_real,k_imag,v_real,v_imag        


class ComplexNetMLP(nn.Module):
    def __init__(
        self,
        hidden_size:int ,
        intermediate_size: int,
        hidden_act: str,
        layer_id :int,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-05,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = ComplexUpLinear(
            hidden_size,
            intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.layer_id = layer_id
        self.down_proj = ComplexLinearBase(
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
       
        if hidden_act != "relu2":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only relu2 is supported for now."
            )
        self.act_fn = ComplexRelu2AndMul()
       
       
        self.ffn_layernorm = ComplexNetRMSNorm(intermediate_size,eps=rms_norm_eps)
       
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:

       # Gate_real,Gate_imag,Up_real,Up_imag = self.gate_up_proj(x_real,x_imag,if_print = (self.layer_id == 0))
        Gate_real,Gate_imag,Up_real,Up_imag = self.gate_up_proj(x_real,x_imag)
        
        x_in_real,x_in_imag = self.act_fn(Gate_real,Gate_imag,Up_real,Up_imag)
       
        x_real,x_imag = self.ffn_layernorm(x_in_real,x_in_imag)
       
        output_real,output_imag = self.down_proj(x_real,x_imag)

        return output_real,output_imag
   
   
class ComplexNetAttention(nn.Module):    
    def __init__(self,
                hidden_size:int,
                num_heads:int,
                num_kv_heads:int,
                layer_id:int,
                rope_theta:int = 10000,
                rope_scaling: Optional[Dict[str, Any]] = None,
                max_position_embeddings: int = 2048,
                quant_config: Optional[QuantizationConfig] = None,
                prefix: str = "",
                rms_norm_eps: float = 1e-05,
                 ) -> None:
        super().__init__()
        
        self.layer_id = layer_id
        
        self.tp_size = 1
        self.hidden_size = hidden_size
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads 
        self.head_dim = hidden_size // num_heads  
              
        self.scaling = (self.head_dim) ** -0.5
       
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
           
        self.qkv_proj = ComplexQKVLinear(
            head_dim=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            prefix=prefix,
        )
        
    
        self.o_proj = ComplexLinearBase(
            in_features=self.head_dim * self.num_heads,
            out_features=hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
       
       
        self.rotary_emb = ComplexGetrope(
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,  
            rope_theta=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim * 2,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        #    quant_config=quant_config,
        #    prefix=add_prefix("attn", prefix),
        )
        
       
        self.attn_layernorm = RMSNormbase(hidden_size * 2,eps=rms_norm_eps)
        
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, dtype=torch.int64) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        
    def forward (
            self,
            positions: torch.Tensor,
            hidden_states_real: torch.Tensor,
            hidden_states_imag: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> Tuple[torch.Tensor,torch.Tensor]:
       
        '''if self.layer_id == 0:
            print(f"attn输入的实部{hidden_states_real}")
            print(f"attn输入的虚部{hidden_states_imag}")'''
        q_real, q_imag, k_real,k_imag, v_real,v_imag= self.qkv_proj(hidden_states_real, hidden_states_imag)
        v_raw = torch.cat([v_real,v_imag],-1)
        q_real = q_real.view(-1, self.num_heads, self.head_dim)
        q_imag = q_imag.view(-1, self.num_heads, self.head_dim)
        k_real = k_real.view(-1, self.num_kv_heads, self.head_dim)
        k_imag = k_imag.view(-1, self.num_kv_heads, self.head_dim)
        v_real = v_real.view(-1, self.num_kv_heads, self.head_dim)
        v_imag = v_imag.view(-1, self.num_kv_heads, self.head_dim)
        position_ids = positions.unsqueeze(0).unsqueeze(0).float()  # [1, 1, num_tokens]
        
        # Reshape inv_freq to match HF: [1, head_dim, 1]
        inv_freq = self.inv_freq[None, :, None].to(positions.device).to(position_ids.dtype)
        
        # Matrix multiplication (align with HF): theta = inv_freq @ position_ids
        theta = (inv_freq @ position_ids).transpose(1, 2).squeeze(0)  # [num_tokens, head_dim]
        
        cos = torch.cos(theta).to(q_real.dtype)
        sin = torch.sin(theta).to(q_real.dtype)
        
        q_real, q_imag = self._apply_rope_complex(q_real, q_imag, cos, sin)
        k_real, k_imag = self._apply_rope_complex(k_real, k_imag, cos, sin)
        
        '''ckpt = torch.load('/home/wangyuxing/q.pt', map_location='cuda')    

        q_real_need = ckpt['q']['real']
        q_imag_need = ckpt['q']['imag']
        q_imag_need= q_imag_need.squeeze(0)
        q_real_need= q_real_need.squeeze(0)
        q_real = q_real.transpose(0,1)
        q_imag = q_imag.transpose(0,1)
        print(f"{q_real.shape}")
        print(f"{q_real_need.shape}")
        print((q_real_need - q_real).abs().mean())
        print((q_imag_need- q_imag).abs().mean())
        import sys
        sys.exit()'''
        # 6. Concatenate [real, imag] for Q, K, V in head_dim dimension
        # cat_q/k/v: [num_tokens, num_heads, 2*head_dim] where dim=-1 is [real, imag]
        cat_q = torch.cat([q_real, q_imag], dim=-1)  # [num_tokens, num_heads, 2*head_dim]
        cat_k = torch.cat([k_real, k_imag], dim=-1)  # [num_tokens, num_kv_heads, 2*head_dim]
        cat_v = torch.cat([v_real, v_imag], dim=-1) 

        q = cat_q.reshape(cat_q.shape[0], -1)  # [num_tokens, num_heads * 2*head_dim]
        k = cat_k.reshape(cat_k.shape[0], -1)  # [num_tokens, num_kv_heads * 2*head_dim]
        v = cat_v.reshape(cat_k.shape[0], -1)
        
        
        attn_output_real_imag = self.attn(q,k,v,forward_batch)
        
        attn_output_real_imag = attn_output_real_imag.view(-1, self.num_heads, 2 * self.head_dim)
        attn_out_real, attn_out_imag = torch.split(attn_output_real_imag, self.head_dim, dim=-1)
        attn_out_real = attn_out_real.reshape(-1, self.num_heads * self.head_dim)
        attn_out_imag = attn_out_imag.reshape(-1, self.num_heads * self.head_dim)
        attn_output_real_imag = torch.cat([attn_out_real, attn_out_imag], dim=-1)
        #print(f"差异为{(attn_output_real_imag - v_raw).abs().mean()}")
        attn_normalized_real_imag = self.attn_layernorm(attn_output_real_imag)
        
        attn_real,attn_imag = torch.chunk( attn_normalized_real_imag , 2, dim=-1)
        
                
        '''ckpt = torch.load('/home/wangyuxing/q.pt', map_location='cuda')    
        q_real_need = ckpt['attn']['real']
        q_imag_need = ckpt['attn']['imag']
        q_imag_need = q_imag_need.squeeze(0)
        q_real_need = q_real_need.squeeze(0)
        real_out = attn_real
        imag_out = attn_imag
        print(q_real_need.shape)
        print((q_real_need - real_out).abs().mean())
        print((q_imag_need - imag_out).abs().mean())
        import sys
        sys.exit()'''

        output_real,output_imag = self.o_proj(attn_real,attn_imag)
        
        '''ckpt = torch.load('/home/wangyuxing/q.pt', map_location='cuda')    
        q_real_need = ckpt['attn']['real']
        q_imag_need = ckpt['attn']['imag']
        q_imag_need = q_imag_need.squeeze(0)
        q_real_need = q_real_need.squeeze(0)
        real_out = output_real
        imag_out = output_imag
        print(q_real_need.shape)
        print("检验输出")
        print((q_real_need[0] - real_out[0]).abs().mean())
        print((q_imag_need[0] - imag_out[0]).abs().mean())
        import sys
        sys.exit()'''
       
        return output_real,output_imag
    def _apply_rope_complex(self, x_real, x_imag, cos, sin):
        """Apply complex RoPE rotation (aligned with HuggingFace _apply_rotary_pos_emb)
        
        Complex rotation: (real + i*imag) * e^{i*theta} = (real + i*imag) * (cos + i*sin)
                         = (real*cos - imag*sin) + i*(real*sin + imag*cos)
        
        Args:
            x_real: [num_tokens, num_heads, head_dim]
            x_imag: [num_tokens, num_heads, head_dim]
            cos, sin: [num_tokens, head_dim]
        
        Returns:
            rotated_real, rotated_imag: each [num_tokens, num_heads, head_dim]
        """
        # Expand cos and sin to match x shape: [num_tokens, 1, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Complex rotation (matching HuggingFace's _apply_rotary_pos_emb)
        rotated_real = x_real * cos - x_imag * sin
        rotated_imag = x_real * sin + x_imag * cos
        
        return rotated_real, rotated_imag
   
   
   
class ComplexNetDecoderLayer(nn.Module):
    def __init__(self,
                   config: Complexconfig,
                   layer_id: int = 0,
                   quant_config: Optional[QuantizationConfig] = None,
                   prefix: str = "",
                   ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.self_attn = ComplexNetAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            rms_norm_eps = config.rms_norm_eps
        )
        self.mlp = ComplexNetMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            rms_norm_eps = config.rms_norm_eps
        )
        self.pre_layernorm = ComplexNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = ComplexNetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )      
       
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states_real: torch.Tensor,
        hidden_states_imag: torch.Tensor,
        forward_batch: ForwardBatch,
        residual_real: Optional[torch.Tensor],
        residual_imag: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        # Self Attention
        if  residual_real is None and residual_imag is None:
            residual_real = hidden_states_real
            residual_imag = hidden_states_imag
            hidden_states_real,hidden_states_imag = self.pre_layernorm(hidden_states_real,hidden_states_imag)
            
        else:
            hidden_states_real,hidden_states_imag,residual_real,residual_imag = self.pre_layernorm(hidden_states_real,hidden_states_imag,residual_real,residual_imag)
       
        hidden_states_real,hidden_states_imag = self.self_attn(
            positions=positions,
            hidden_states_real=hidden_states_real,
            hidden_states_imag=hidden_states_imag,
            forward_batch=forward_batch,
        )

        hidden_states_real,hidden_states_imag,residual_real,residual_imag = self.post_layernorm(hidden_states_real,hidden_states_imag,residual_real,residual_imag)
        
        hidden_states_real,hidden_states_imag = self.mlp(
            hidden_states_real,
            hidden_states_imag,
        )
        '''if self.layer_id == 0:
            print(f"第一层的mlp实部输出为{hidden_states_real}")
            print(f"第一层的mlp虚部输出为{hidden_states_imag}")'''
            
            
        return hidden_states_real,hidden_states_imag, residual_real,residual_imag
   
class ComplexNetLMBase(nn.Module):
    def __init__(
        self,
        config: Complexconfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type:type[nn.Module]=ComplexNetDecoderLayer,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # self.pp_group = get_pp_group()
       
        # if self.pp_group.is_first.rank:
        self.token_embeddings_real = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix="token_embeddings_real",
        )
        self.token_embeddings_imag = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix="token_embeddings_imag",
        )
           
           
        # else:
        #     self.embed_tokens_real = PPMissingLayer()
        #     self.embed_tokens_imag = PPMissingLayer()
       
        decoder_layer_type = decoder_layer_type or ComplexNetDecoderLayer
       
        self.layer = make_layers(
            config.num_hidden_layers,
            lambda idx,prefix: decoder_layer_type(
                layer_id = idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            # pp_rank=self.pp_group.rank_in_group,
            # pp_size=self.pp_group.world_size,
            prefix=add_prefix("layer", prefix),
        )
       
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
       
       
       
       
        # if self.pp_group.is_last.rank:
        self.final_norm = ComplexNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
       
        # else:
        #     self.final_norm = PPMissingLayer()
       
    def get_input_embeddings(self) -> Tuple[nn.Embedding,nn.Embedding]:    
        return self.token_embeddings_real,self.token_embeddings_imag
       
    def get_input_embedding(self,input_ids: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        input_embeds_real = self.token_embeddings_real(input_ids)
        input_embeds_imag = self.token_embeddings_imag(input_ids)
       
        if hasattr(self.config, "scale_emb"):
            return self.config.scale_emb * input_embeds_real,self.config.scale_emb * input_embeds_imag
       
        else:
            return input_embeds_real,input_embeds_imag
       
       
           
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds_real: Optional[torch.Tensor] = None,
        input_embeds_imag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds_real is None or input_embeds_imag is None:
#            print(f"input_id 为{input_ids}")
            hidden_state_real, hidden_state_imag = self.get_input_embedding(input_ids)
        else:
            hidden_state_real = input_embeds_real
            hidden_state_imag = input_embeds_imag
        residual_real = None
        residual_imag = None
               
#        print(f"model输入hidden_states实部{hidden_state_real }")
#        print(f"model输入hidden_states实部{hidden_state_imag }")    
           
        for i in range(self.start_layer,self.end_layer):
            layer = self.layer[i]
            hidden_state_real,hidden_state_imag, residual_real, residual_imag = layer(
                positions=positions,
                hidden_states_real=hidden_state_real,
                hidden_states_imag=hidden_state_imag,
                forward_batch=forward_batch,
                residual_real=residual_real,
                residual_imag=residual_imag,
            )
           
        # if not self.pp_group.is_last.rank:
        #     return PPProxyTensors(
        #         {
        #         "hidden_state_real":hidden_state_real,
        #         "hidden_state_imag":hidden_state_imag,
        #         "residual_real":residual_real,
        #         "residual_imag":residual_imag,
        #         }
        #     )
           
           
        # else:
        hidden_state_real,hidden_state_imag,_,__ = self.final_norm(hidden_state_real,hidden_state_imag,residual_real,residual_imag)
        hidden_state = torch.cat([hidden_state_real, hidden_state_imag], dim=-1)
        return hidden_state
       
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            self.config.num_hidden_layers,
            self.config.model_type,
        ):
            if not isinstance(self.layer[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )
               
               
class ComplexNetLM(nn.Module):      
    default_bitsandbytes_target_modules = [
         ".gate_proj.",
         ".down_proj.",
         ".up_proj.",
         ".q_proj.",
         ".k_proj.",
         ".v_proj.",
         ".o_proj.",
     ]
    bitsandbytes_stacked_params_mapping = {
         # shard_name, weight_name, index
         "q_proj": ("qkv_proj", 0),
         "k_proj": ("qkv_proj", 1),
         "v_proj": ("qkv_proj", 2),
         "gate_proj": ("gate_up_proj", 0),
         "up_proj": ("gate_up_proj", 1),
     }   
    def __init__(
        self,
        config: Complexconfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    )->None:
        super().__init__()
           
        # self.pp_group = get_pp_group()
        self.config = config
       
        self.config.hidden_size = self.config.hidden_size // 2
       
        self.quant_config = quant_config
   
   
       
        self.model = ComplexNetLMBase(
            config=self.config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=ComplexNetDecoderLayer,
        )
   
        # if self.pp_group.is_last_rank:  
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            self.config.hidden_size * 2,
            prefix="lm_head",
    )
        # else:
        #     self.lm_head = PPMissingLayer()
   
       
        self.logits_processor = LogitsProcessor(self.config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
   
   
   
    def get_input_embeddings(self) -> Tuple[nn.Embedding,nn.Embedding]:    
        return self.model.get_input_embeddings()
   
   
   
    def get_input_embedding(self,input_ids: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        return self.model.get_input_embedding(input_ids)
   
   
   
   
    @torch.no_grad()
    def forward(
        self,
        input_ids:torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds_real: Optional[torch.Tensor] = None,
        input_embeds_imag: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        # pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
       
       
       
       
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds_real=input_embeds_real,
            input_embeds_imag=input_embeds_imag,
        #    pp_proxy_tensors=pp_proxy_tensors,
            )
        '''ckpt = torch.load('/home/wangyuxing/q.pt',map_location='cuda')
        out_real,out_imag = ckpt['outputs']["real"], ckpt['outputs']["imag"]
        out = torch.cat([out_real,out_imag],dim=-1)
        out = out.squeeze(0)
        print(out.shape)
        print(f"最终model第一个token输出差异为{(out[0]-hidden_states[0]).abs().mean()}")
        print(f"最终model第二个token输出差异为{(out[1]-hidden_states[1]).abs().mean()}")
        import sys
        sys.exit()'''
        if not get_embedding:
            logits = self.logits_processor(
                input_ids,hidden_states,self.lm_head,forward_batch
            )
            return logits
        else:
            return self.pooler(hidden_states,forward_batch)
           
    @property
    def start_layer(self):
        return self.model.start_layer
    @property
    def end_layer(self):
        return self.model.end_layer
   
    def get_embed_and_head(self):
        return self.model.embed_tokens_real.weight,self.model.embed_tokens_imag.weight, self.lm_head.weight
   
    def set_embed_and_head(self, embed_real,embed_imag, head):
        del self.model.token_embeddings_real.weight
        del self.model.token_embeddings_imag.weight
        del self.lm_head.weight
        self.model.token_embeddings_real.weight = embed_real
        self.model.token_embeddings_imag.weight = embed_imag
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
   
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)
   
   
   
    # ... 在您的主模型类中 ...
    '''def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Custom weight loader for Single-GPU inference.
        Handles parameter renaming, tensor fusion (QKV/GateUp), and weight tying.
        """
        # 1. 定义映射表：源参数名片段 -> (目标融合参数名片段, 分片ID)
        stacked_params_mapping = [
            # (target_param_name, source_weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
       
        print("--- [ComplexNet] Starting Single-GPU Custom Weight Loading ---")
        i = 0
        for name in tqdm(params_dict):
            print(f"模型的第{i}个参数名为{name}")
            i+=1
            if i > 100:
                break
       
        for name, loaded_weight in tqdm(weights, desc="Loading Weights"):
           
           
            if name != "lm_head.weight":
                name = "model." + name
           
            # 2. 跳过无关参数
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
           
            # 3. 处理权重绑定 (Weight Tying)
            # 在单卡模式下，init 中已经执行了 self.lm_head.weight = self.model.embed_tokens.weight
            # 所以当加载到 "lm_head.weight" 时，我们直接跳过，避免重复加载或覆盖
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            # --- 核心加载逻辑：处理融合层 (Fused Layers) ---
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # 检查当前权重是否属于某个融合组 (e.g. "q_proj" in "model...self_attn.q_proj.weight")
                if weight_name not in name:
                    continue
               
                # 构造目标参数名 (e.g. "model...self_attn.qkv_proj.weight")
                row_target_name = name.replace(weight_name, param_name)
               
                # 跳过 GPTQ 的额外 bias
                if row_target_name.endswith(".bias") and row_target_name not in params_dict:
                    continue
               
                target_name = row_target_name + ".weight"
               
                if target_name in params_dict:
                    param = params_dict[target_name]
                    # 调用 SGLang 参数对象自带的加载器，指定分片ID
                    # 即使是单卡，这个 loader 也能正确处理拼接逻辑
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    is_stacked = True
                    break
           
            if is_stacked:
                continue
            # --- 核心加载逻辑：处理普通层 (Normal Layers) ---
            # e.g., embed_tokens, layernorm, o_proj, down_proj
           
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name in params_dict:
                param = params_dict[name]
                # 使用默认加载器直接拷贝权重
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
               
               
            else:
                deal_name = name + ".weight"
                if deal_name in params_dict:
                    param = params_dict[deal_name]
                    weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:    
                # 这是一个预期的警告，如果源文件包含一些模型不需要的参数
                # logger.warning(f"Parameter {name} not found in params_dict")
                    print(f" [WARNING] 权重未匹配到 SGLang 模型: {name}")
       
        print("--- [ComplexNet] Weight Loading Finished ---")'''
        
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        print("--- [ComplexNet] Starting Single-GPU Custom Weight Loading ---")
        '''i = 0
        for name in tqdm(params_dict):
            print(f"模型的第{i}个参数名为{name}")
            i+=1
            if i > 100:
                break'''
        for name, loaded_weight in tqdm(weights, desc="Loading Weights"):
            if name in params_dict:
                param = params_dict[name]
                    # 使用默认加载器直接拷贝权重
                weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                weight_loader(param, loaded_weight)
            else :
                print(f" [WARNING] 权重未匹配到 SGLang 模型: {name}")
        print("--- [ComplexNet] Weight Loading Finished ---")
EntryClass = ComplexNetLM