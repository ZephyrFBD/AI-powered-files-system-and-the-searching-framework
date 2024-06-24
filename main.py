from collections import OrderedDict
#from transformers import pipeline
from tdt import read_txt_files

#遍历txt：test文件夹下所有子文件夹
directory = r'.\test'
out = read_txt_files(directory)
print(out.keys())
#out为字典,(directory,content)
input = list(out.values())
print(input)