import os

def read_txt_files(directory):
    """
    遍历指定目录及其所有子目录中的所有 .txt 文件，并返回它们的内容。
    
    参数：
    directory (str): 要遍历的目录路径。
    
    返回：
    list: 包含所有 .txt 文件内容的列表。
    """
    dic = {}
    n = 0

    # 使用 os.walk 遍历所有子目录和文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    # 处理文件内容
                    dic[file_path] = content
                    n += 1
    # 输出结果
    #print(dic)
    #print(list(dic.values()))
    return dic
"""
# 示例调用
directory = r'.\test'
out = read_txt_files(directory)
print(out)
"""