import os
import pandas as pd
from gensim.models import Doc2Vec
import umap
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed

# 加载 CSV 文件
csv_path = './your_csv_file.csv'
df = pd.read_csv(csv_path)

# 获取标签（使用文件路径作为标签）
labels = df['File Path'].tolist()

# 为标签创建颜色字典
unique_labels = list(set(labels))
color_palette = sns.color_palette("hsv", len(unique_labels))
label_to_color = {label: color_palette[i] for i, label in enumerate(unique_labels)}

# 根据标签分配颜色
colors = [label_to_color[label] for label in labels]

# 获取所有模型的文件夹路径
model_folder = './models'
model_files = [f for f in os.listdir(model_folder) if f.startswith('doc2vec_model')]

# 创建保存UMAP可视化图像的文件夹
output_folder = './umap_visualizations'
os.makedirs(output_folder, exist_ok=True)

def process_model(model_file):
    model_path = os.path.join(model_folder, model_file)
    
    # 加载 Doc2Vec 模型
    model = Doc2Vec.load(model_path)

    # 获取所有文档向量并转换为 NumPy 数组
    doc_vectors = []
    num_documents = len(model.dv)
    for i in range(num_documents):
        doc_vectors.append(model.dv[i])
    doc_vectors = np.array(doc_vectors)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_vectors = umap_model.fit_transform(doc_vectors)

    # 绘制带颜色区分的 UMAP 可视化图
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1], c=colors, marker='.')
    plt.title(f'UMAP Visualization of {model_file}')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # 保存图像
    output_image_path = os.path.join(output_folder, f'{model_file}.png')
    plt.savefig(output_image_path)
    plt.close()

    return f'Finished processing {model_file}'

# 并行处理模型文件
num_cores = 20
results = Parallel(n_jobs=num_cores)(
    delayed(process_model)(model_file) for model_file in tqdm(model_files, desc='Total Progress')
)

# 打印处理结果
for result in results:
    print(result)

print('All models visualized and saved successfully.')
