from datasets import load_dataset

# 加载数据集
ds = load_dataset("meliascosta/wiki_academic_subjects")

# 保存数据集到本地
save_path = "/path/to/save/directory"  # 替换为你想要保存数据集的目录路径

# 存储数据集
ds.save_to_disk(save_path)

# 打印成功保存的消息
print(f"Dataset saved to {save_path}")
