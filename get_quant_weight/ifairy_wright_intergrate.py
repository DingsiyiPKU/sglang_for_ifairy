import os
import tqdm
# 【关键】强制使用 CPU，避免加载 PyTorch CUDA 时卡死
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors.torch import load_file, save_file
import torch.nn as nn

def main():
    real = "_real"
    imag = "_imag"
    attn = "attn_"
    len = 5
    norm = 'norm'
    input_path = ["/home/wangyuxing/trans-ifairy-full-1.3B/model-00001-of-00002.safetensors","/home/wangyuxing/trans-ifairy-full-1.3B/model-00002-of-00002.safetensors"]                   #需要转化的模型路径
    output_path = "/home/wangyuxing/sglang-Fairy-plus-minus-i-1.3B/full_model_1.3B_sglang.safetensors"        #转化后模型保存路径
    
    stacked_params_mapping = [
            # (target_param_name, source_weight_name, shard_id)
            ("qkv_proj.q_weight", "q_proj"),
            ("qkv_proj.k_weight", "k_proj"),
            ("qkv_proj.v_weight", "v_proj"),
            ("gate_up_proj.gate_weight", "gate_proj"),
            ("gate_up_proj.up_weight", "up_proj"),
        ]
    
    

    new_state_dict = {}
    try:
        for path in input_path:
            print(f"正在加载原始权重: {path}")
        # 加载文件
            state_dict = load_file(path, device="cpu")
            
            
            # 遍历所有权重
            for name, tensor in tqdm.tqdm(state_dict.items()):           
                if norm in name :          
                    if name.endswith(real) :
                        real_tensor = tensor
                        base_name = name[:-len]
                        imag_name = base_name + imag
                        if imag_name not in state_dict:
                            print(f"警告: 找不到 {name} 对应的虚部，将跳过此层！")
                            continue
                        
                        else:
                            imag_tensor = state_dict[imag_name]                    
                            complex_tensor = torch.cat([real_tensor, imag_tensor], dim=0)
                            if attn not in name:
                                using_name = "model." + base_name +  '.weight'  
                            else:
                                using_name = "model." + base_name
                            new_state_dict[using_name] = complex_tensor
                        
                            print(f"处理复数层: {using_name} | 形状: {complex_tensor.shape}")
                    elif name.endswith(imag) :
                        continue
                    else :
                        print(f"{name}无法被处理")
                        

                else:
                    if name != "lm_head.weight":
                        name = "model." + name
                        if name != "model.token_embeddings_imag.weight" and name != "model.token_embeddings_real.weight":
                            name = name + '.weight'       
                    for param_name, weight_name in stacked_params_mapping:
                        # 检查当前权重是否属于某个融合组 (e.g. "q_proj" in "model...self_attn.q_proj.weight")
                        if weight_name not in name:
                            continue
                    
                        # 构造目标参数名 (e.g. "model...self_attn.qkv_proj.weight")
                        else :
                            name = name.replace(weight_name, param_name)
                            break
                    print(f"处理复数层: {name} | 形状: {tensor.shape}")
                    new_state_dict[name] = tensor

    except Exception as e:
        print(f"发生错误: {e}")
        
        
    try:
        print(f"处理完成，正在保存到: {output_path}")
        save_file(new_state_dict, output_path)
        print("成功！")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()