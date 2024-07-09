from gensim.models import Doc2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 加载 Doc2Vec 模型
model_path = './doc2vec_model'
model = Doc2Vec.load(model_path)

# 获取所有文档向量并转换为 NumPy 数组，添加进度条显示
doc_vectors = []
num_documents = len(model.docvecs)
with tqdm(total=num_documents, desc='Extracting document vectors') as pbar:
    for i in range(num_documents):
        doc_vectors.append(model.docvecs[i])
        pbar.update(1)  # 更新进度条

doc_vectors = np.array(doc_vectors)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=42, n_jobs=-1)

# t-SNE 转换，添加进度条显示
with tqdm(total=len(doc_vectors), desc='Applying t-SNE') as pbar:
    tsne_vectors = tsne.fit_transform(doc_vectors)
    pbar.update(len(doc_vectors))  # 更新进度条

# 绘制 t-SNE 可视化图
plt.figure(figsize=(10, 8))
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], marker='.')
plt.title('t-SNE Visualization of Document Vectors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
