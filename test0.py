from transformers import pipeline

# 加载文本分类的 pipeline
classifier = pipeline("text-classification")

# 准备待分类的文本
texts = [
    "这是一篇关于文学的文章。",
    "数学是一门重要的学科，它涉及到数字和运算。",
    "历史学研究人类社会过去的事件和行为。",
]

# 进行文本分类
for text in texts:
    result = classifier(text)
    print(f"文本: {text}")
    print(f"预测的学科: {result[0]['label']}, 置信度: {result[0]['score']:.4f}")
    print()
