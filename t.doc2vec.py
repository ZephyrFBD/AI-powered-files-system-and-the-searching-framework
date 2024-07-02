import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import csv
import logging
from tqdm import tqdm

# 下载必要的nltk资源
nltk.download('punkt')
nltk.download('stopwords')

logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start training Doc2Vec model')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing punctuation, and stopwords."""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]  # Remove punctuation and numbers
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words

def read_paragraphs_from_csv(csv_file):
    """Read paragraphs from a CSV file."""
    paragraphs = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            paragraphs.append(row[2])  # Paragraph text is in the third column
    return paragraphs

class ProgressLogger:
    def __init__(self, total):
        self.progress_bar = tqdm(total=total, desc="Training Progress", unit="epoch")
        self.epoch = 0

    def on_epoch_end(self, *args):
        self.epoch += 1
        self.progress_bar.update(1)

def train_doc2vec_model(paragraphs):
    """Train a Doc2Vec model on the given paragraphs."""
    documents = [TaggedDocument(words=preprocess_text(paragraph), tags=[i]) for i, paragraph in enumerate(paragraphs)]
    
    model = doc2vec.Doc2Vec(vector_size=100, window=5, min_count=2, workers=8, epochs=40)
    model.build_vocab(documents)
    logging.info('Vocabulary built.')

    total_epochs = 40
    progress_logger = ProgressLogger(total=total_epochs)

    for epoch in range(total_epochs):
        model.train(documents, total_examples=model.corpus_count, epochs=1)
        progress_logger.on_epoch_end()
        model.alpha -= 0.006  # Decrease the learning rate
        model.min_alpha = model.alpha  # Fix the learning rate, no decay

    progress_logger.progress_bar.close()
    logging.info('Model trained.')
    
    return model

def save_model(model, model_path):
    """Save the trained Doc2Vec model."""
    model.save(model_path)
    logging.info(f'Model saved as {model_path}')

if __name__ == '__main__':
    input_csv = './text_files.csv'
    model_path = './doc2vec_model'
    
    logging.info(f'Reading paragraphs from {input_csv}')
    paragraphs = read_paragraphs_from_csv(input_csv)
    
    logging.info('Training Doc2Vec model')
    model = train_doc2vec_model(paragraphs)
    
    save_model(model, model_path)
    
    logging.info('Finished training Doc2Vec model')
    print(f"Trained Doc2Vec model saved as '{model_path}'.")
