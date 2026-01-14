import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file 
import sys
import os
SPECIFIED_GPU = "1"  
os.environ["CUDA_VISIBLE_DEVICES"] = SPECIFIED_GPU
def inference():
    
    if "modeling_Fairy_plus_minus_i" in sys.modules:
        del sys.modules["modeling_Fairy_plus_minus_i"]
    
    # 1. 定义 checkpoint 路径
    ckpt_path = "/home/wangyuxing/pure_Fairy-plus-minus-i-700M"
    save_path = "/home/wangyuxing/Fairy-plus-minus-i-700M-quantized.safetensors"
    
    # 2. 确定设备：强制使用单卡或 CPU，不涉及 DataParallel 或分布式
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. 加载 Tokenizer
    # Tokenizer 负责将文本转为 Transformer 能够理解的 input_ids
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

    # 4. 加载模型
    # 我们不使用 device_map="auto"，而是手动指定 device，确保推理逻辑在单路执行
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, 
        torch_dtype=torch.bfloat16 ,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    
    # 设置为评估模式（关闭 Dropout 等）
    model.eval()

    # 5. 准备输入
    prompt ="as we"
    # return_tensors="pt" 返回的是 PyTorch 张量
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"PAD Token 内容: {tokenizer.pad_token}")
    print(f"EOS Token 内容: {tokenizer.eos_token}")
    print(f"PAD Token ID: {tokenizer.pad_token_id}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")

    # 6. 执行推理（生成）
    # 这里是 Transformer 的核心过程：
    # Pre-fill (计算 Prompt 的 KV Cache) + Decoding (逐个 Token 生成)
    print("Generating...")
    with torch.no_grad():  # 推理阶段关闭梯度计算，节省内存
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,    # 最大生成长度
            do_sample=True,        # 是否采样
            temperature=0.6,       # 控制随机性
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id
        )

    # 7. 解码结果
    # output_ids 包含输入的 prompt 和生成的 token
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("-" * 20)
    print(f"Input: {prompt}")
    print(f"Output: {response}")

    
    print(f"Saving quantized model to {save_path}...")
    
    # 获取 state_dict
    state_dict = model.state_dict()
    
    # 关键：将 tensor 移回 CPU 并确保内存连续，避免保存出错
    # 同时过滤掉不需要保存的缓存或辅助参数（如果有的话）
    state_dict_to_save = {k: v.cpu().contiguous() for k, v in state_dict.items()}
    
    # 使用 safetensors 保存
    save_file(state_dict_to_save, save_path)
    
    print("Success: Quantized model saved successfully!")
    
    
if __name__ == "__main__":
    inference()