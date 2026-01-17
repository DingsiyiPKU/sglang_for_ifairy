import os
import torch
import sglang as sgl
import asyncio
import uvloop


print("sglang path:", sgl.__file__)

def main():
    model_path = "/home/wangyuxing/sglang-Fairy-plus-minus-i-1.3B"
    print(f"正在从{model_path}加载模型")
    llm = sgl.Engine(
    model_path=model_path,
    tokenizer_path=model_path,
    trust_remote_code=True,
    context_length=2048,
    max_running_requests=8,
    mem_fraction_static=0.6,
    
    dtype="bfloat16", 
    attention_backend="triton",  
    disable_cuda_graph=True,
)

    prompt ="as we"
    sampling_params = {
    "temperature": 0.6,
    "top_p": 0.90,
    "max_new_tokens": 5,
    #"stop_token_ids": [2],
    }


    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]

    print("===============================")
    print("===============================")
    print("Prompt:", prompt)
    print("Generated text:", output["text"])

if __name__ == "__main__":
    main()