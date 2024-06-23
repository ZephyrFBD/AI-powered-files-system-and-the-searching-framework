import os
from glob import glob
from collections import OrderedDict
from transformers import pipeline

n=0
dic = OrderedDict()


#遍历所有目录test下的txt
directory = r'.\test'
for file_path in glob(os.path.join(directory, '*.txt')):
    with open(file_path, 'r') as file:
        content = file.read()
        # 处理文件内容
        #print(f'Reading: {file_path}')
        #print(content,"\n",n)
        n=n+1
        dic[file_path] = content
print(dic)
print(list(dic.values()))
out = list(dic.values())

# 执行情感分析
classifier = pipeline("sentiment-analysis", model="./model")
#save_path = r".\model"
#classifier.save_pretrained(save_path)
results = classifier(list(dic.values()))

# 迭代结果和文件名
for file_path, result in zip(dic.keys(), results):
    print(f"File: {file_path}")
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    print()