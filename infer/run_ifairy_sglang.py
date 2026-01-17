import os
import json
import asyncio
import uvloop
import nest_asyncio
import multiprocessing
import torch
from datetime import datetime

# 大模型框架组件
import sglang as sgl
import lm_eval
from lm_eval.api.registry import get_model
from lm_eval.utils import make_table

# 1. 定义你要评测的任务列表和重复次数
ALL_TASKS = ["arc_easy", "arc_challenge", "boolq", "openbookqa", "winogrande", "hellaswag"]
NUM_RUNS = 1
BASE_SEED = 42 # 定义一个基础种子，每次运行递增

def worker(gpu_id, job_queue, file_locks):
    """
    工作进程函数：
    - 绑定到一张 GPU。
    - 只初始化一次 SGLang Engine。
    - 不断从队列中获取并执行评测作业。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    nest_asyncio.apply()
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.get_event_loop = lambda: loop

    print(f"--- [GPU {gpu_id}] Worker 启动，正在初始化 SGLang Engine... ---")
    
    model_path = "/home/wangyuxing/sglang_model/sglang-Fairy-plus-minus-i-1.3B"
    print(f"测试{model_path} ing ...")
    model_obj = get_model("sglang")(
        pretrained=model_path,
        tokenizer_path=model_path,
        trust_remote_code=True,
        tp_size=1,
        dp_size=1, 
        dtype="bfloat16",
        attention_backend="triton",
        disable_cuda_graph=True,
        mem_fraction_static=0.75,
        context_length=2048
    )
    print(f"--- [GPU {gpu_id}] Engine 初始化完成，等待作业... ---")

    while True:
        job = job_queue.get()
        if job is None:
            break
            
        # --- 核心修改 1: 从作业中解包出种子 ---
        task_name, run_id, seed = job

        print(f"--- [GPU {gpu_id}] 开始处理作业: {task_name} (第 {run_id}/{NUM_RUNS} 次, 种子: {seed}) ---")
        
        # --- 核心修改 2: 将种子传递给 simple_evaluate ---
        results = lm_eval.simple_evaluate(
            model=model_obj,
            tasks=[task_name],
            batch_size=128,
            # lm-eval 会自动处理 numpy 和 torch 的种子设置
            random_seed=seed 
        )

        if results:
            output_dir = os.path.join("/home/wangyuxing/eval-result/1300M_bf16_eval_results", task_name)
            os.makedirs(output_dir, exist_ok=True)
            history_path = os.path.join(output_dir, "results_history.json")
            
            lock = file_locks[task_name]
            with lock:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_metrics = results["results"][task_name]

                if os.path.exists(history_path):
                    try:
                        with open(history_path, "r", encoding="utf-8") as f:
                            history_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        history_data = []
                else:
                    history_data = []
                
                # --- 核心修改 3: 在保存结果时也记录下种子 ---
                history_data.append({
                    "run_id": run_id,
                    "seed": seed, # 记录本次运行使用的种子
                    "timestamp": timestamp,
                    "metrics": current_metrics
                })
                
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=4)
            
            print(f"--- [GPU {gpu_id}] 作业 {task_name} (第 {run_id} 次) 完成！---")
            
    print(f"--- [GPU {gpu_id}] 所有作业完成，Worker 退出。 ---")


def main():
    num_gpus = torch.cuda.device_count()
    if not num_gpus:
        print("错误：未检测到任何可用的 GPU。")
        return
        
    print(f"检测到 {num_gpus} 张可用 GPU。")

    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    file_locks = {task: manager.Lock() for task in ALL_TASKS}
    job_queue = manager.Queue()
    total_jobs = 0

    # --- 核心修改 4: 在创建作业时生成并加入种子 ---
    for run_id in range(1, NUM_RUNS + 1):
        # 为每一轮实验生成一个不同的种子
        current_seed = BASE_SEED + run_id - 1 
        for task_name in ALL_TASKS:
            job_queue.put((task_name, run_id, current_seed))
            total_jobs += 1
            
    for _ in range(num_gpus):
        job_queue.put(None)

    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(gpu_id, job_queue, file_locks))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\n--- 所有 {total_jobs} 个评测作业已全部完成！---")

if __name__ == "__main__":
    # 强制开启数据集脚本信任
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    main()