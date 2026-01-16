# 获取量化后参数的方法：
1. 将文件中的modeling_Fairy_plus_minus_i.py替换hugging face中的同名文件
2. 设置transformer_infer.py中的ckpt路径和量化后模型参数保存路径
3. 运行transformer_infer.py
4. 设置ifairy_wright_intergrate.py中模型路径和转化后路径
5. 运行ifairy_wright_intergrate.py
6. 删除model.safetensors.index.json文件（如果有）
