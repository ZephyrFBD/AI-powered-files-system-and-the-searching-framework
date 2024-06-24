# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="google-bert/bert-base-uncased")
save_path = r".\model"
pipe.save_pretrained(save_path)