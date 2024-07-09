import os
import csv
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# 下载必要的nltk资源
nltk.download('punkt')
nltk.download('stopwords')

# 停用词和标点符号
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    """文本预处理，包括转换为小写、去除标点符号和停用词等处理。"""
    text = text.lower()  # 转换为小写
    words = word_tokenize(text)  # 分词

    # 去除标点符号和停用词
    words = [word for word in words if word not in stop_words and word not in punctuation]
    
    return ' '.join(words)

def preprocess_and_save_to_csv(input_csv, output_csv):
    """读取原始文本数据，进行预处理，并保存到新的CSV文件中。"""
    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取头部信息

        with open(output_csv, 'w', newline='', encoding='utf-8') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(header + ['Processed Text'])  # 添加新的列名

            for row in tqdm(reader, desc="Processing rows", unit="row"):
                original_text = row[2]  # 假设文本在第三列
                processed_text = preprocess_text(original_text)
                writer.writerow(row + [processed_text])  # 将处理后的文本添加到新的CSV文件中

if __name__ == '__main__':
    input_csv = './text_files.csv'  # 输入原始文本数据的CSV文件路径
    output_csv = './processed_text_files.csv'  # 输出预处理后的文本数据的CSV文件路径
    
    preprocess_and_save_to_csv(input_csv, output_csv)
    
    print(f"Preprocessed text saved to '{output_csv}'.")
