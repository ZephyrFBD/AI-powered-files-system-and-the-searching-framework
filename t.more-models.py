import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import csv
import logging
from tqdm import tqdm
from itertools import product

# 下载必要的nltk资源
nltk.download('punkt')
nltk.download('stopwords')

# 设置日志记录
logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start training Doc2Vec models with parameter cross-analysis')

# 参数网格定义
vector_sizes = [50, 100, 200]
window_sizes = [3, 5, 7]
min_counts = [1, 2, 3]
epochs = 20  # 每个模型的训练轮数

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """文本预处理，包括转换为小写、去除标点符号和停用词等处理。"""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]  # 去除标点符号和数字
    words = [word for word in words if word not in stop_words]  # 去除停用词
    return words

def read_paragraphs_from_csv(csv_file):
    """从CSV文件中读取段落。"""
    paragraphs = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过头部行
        for row in reader:
            paragraphs.append(row[2])  # 段落文本位于第三列
    return paragraphs

class ProgressLogger:
    def __init__(self, total):
        self.progress_bar = tqdm(total=total, desc="Training Progress", unit="epoch")
        self.epoch = 0

    def on_epoch_end(self, *args):
        self.epoch += 1
        self.progress_bar.update(1)

def train_doc2vec_model(paragraphs, vector_size, window, min_count):
    """训练Doc2Vec模型。"""
    documents = [TaggedDocument(words=preprocess_text(paragraph), tags=[i]) for i, paragraph in enumerate(paragraphs)]
    
    model = doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=8, epochs=epochs)
    model.build_vocab(documents)
    logging.info(f'Vocabulary built for model with vector_size={vector_size}, window={window}, min_count={min_count}')

    total_epochs = epochs
    progress_logger = ProgressLogger(total=total_epochs)

    for epoch in range(total_epochs):
        model.train(documents, total_examples=model.corpus_count, epochs=1)
        progress_logger.on_epoch_end()
        model.alpha -= 0.002  # 降低学习率
        model.min_alpha = model.alpha  # 固定学习率，无衰减

    progress_logger.progress_bar.close()
    logging.info(f'Model trained for vector_size={vector_size}, window={window}, min_count={min_count}')
    
    return model

def save_model(model, model_path):
    """保存训练好的Doc2Vec模型。"""
    model.save(model_path)
    logging.info(f'Model saved as {model_path}')

if __name__ == '__main__':
    input_csv = './text_files.csv'
    logging.info(f'Reading paragraphs from {input_csv}')
    paragraphs = read_paragraphs_from_csv(input_csv)

    # 交叉分析所有参数组合
    parameter_combinations = list(product(vector_sizes, window_sizes, min_counts))

    for vector_size, window_size, min_count in parameter_combinations:
        model_name = f'doc2vec_model_vs{vector_size}_ws{window_size}_mc{min_count}'
        model_path = f'./models/{model_name}'
        
        logging.info(f'Training Doc2Vec model with vector_size={vector_size}, window={window_size}, min_count={min_count}')
        model = train_doc2vec_model(paragraphs, vector_size, window_size, min_count)
        
        save_model(model, model_path)
        
        logging.info(f'Finished training and saved Doc2Vec model: {model_name}')
    
    logging.info('Finished training all Doc2Vec models with parameter cross-analysis')
    print('All models trained and saved successfully.')
