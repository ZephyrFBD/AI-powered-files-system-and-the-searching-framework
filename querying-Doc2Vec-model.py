from gensim.models import doc2vec
import logging
import csv

# Setup logging
logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start querying Doc2Vec model')

def read_paragraphs_from_csv(csv_file):
    """Read paragraphs from a CSV file."""
    paragraphs = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            paragraphs.append(row[0])  # Paragraph text is in the third column
    return paragraphs

def find_similar_sentences(model, input_sentence, paragraphs):
    """Find the most similar sentences to the input sentence using the Doc2Vec model."""
    input_vector = model.infer_vector(input_sentence.split())
    most_similar = model.dv.most_similar([input_vector], topn=10)
    similar_sentences = [(paragraphs[i], similarity) for i, similarity in most_similar]
    return similar_sentences

if __name__ == '__main__':
    input_csv = './processed_text_files.csv'
    model_path = './doc2vec_model'
    
    logging.info(f'Loading Doc2Vec model from {model_path}')
    model = doc2vec.Doc2Vec.load(model_path)
    
    logging.info(f'Reading paragraphs from {input_csv}')
    paragraphs = read_paragraphs_from_csv(input_csv)
    
    input_sentence = input("Enter a sentence: ")
    
    logging.info(f'Finding similar sentences to: "{input_sentence}"')
    similar_sentences = find_similar_sentences(model, input_sentence, paragraphs)
    
    print("Most similar sentences:")
    for sentence, similarity in similar_sentences:
        print(f"Path: {sentence}, Similarity: {similarity:.4f}")
    
    logging.info('Finished querying Doc2Vec model')
