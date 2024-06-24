from collections import OrderedDict
from transformers import pipeline
from tdt import read_txt_files

#遍历txt：test文件夹下所有子文件夹
directory = r'.\test'
out = read_txt_files(directory)
print(out)
#out为字典,(directory,content)
input = list(out.values())

# 执行情感分析
classifier = pipeline("sentiment-analysis")
#save_path = r".\model"
#classifier.save_pretrained(save_path)
results = classifier(input)

# 迭代结果和文件名
for file_path, result in zip(out.keys(), results):
    print(f"File: {file_path}")
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    print()