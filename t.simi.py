import gensim.models.doc2vec as doc2vec
import logging
import csv
import os

# Setup logging
logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start querying Doc2Vec model')

def read_paragraphs_and_paths_from_csv(csv_file):
    """Read paragraphs and file paths from a CSV file."""
    paragraphs = []
    file_paths = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            file_paths.append(row[0])  # File path is in the first column
            paragraphs.append(row[2])  # Paragraph text is in the third column
    return paragraphs, file_paths

def find_similar_sentences(model, input_sentence, paragraphs, file_paths):
    """Find the most similar sentences to the input sentence using the Doc2Vec model."""
    input_vector = model.infer_vector(input_sentence.split())
    most_similar = model.dv.most_similar([input_vector], topn=10)
    similar_sentences = [(paragraphs[i], similarity, file_paths[i]) for i, similarity in most_similar]
    return similar_sentences

if __name__ == '__main__':
    input_csv = './text_files.csv'
    model_path = './doc2vec_model'
    
    logging.info(f'Loading Doc2Vec model from {model_path}')
    model = doc2vec.Doc2Vec.load(model_path)
    
    logging.info(f'Reading paragraphs and paths from {input_csv}')
    paragraphs, file_paths = read_paragraphs_and_paths_from_csv(input_csv)
    
    input_sentence = input("Enter a sentence: ")
    
    logging.info(f'Finding similar sentences to: "{input_sentence}"')
    similar_sentences = find_similar_sentences(model, input_sentence, paragraphs, file_paths)
    
    print("Most similar sentences:")
    for sentence, similarity, file_path in similar_sentences:
        print(f"Sentence: {sentence}, Similarity: {similarity:.4f}")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                print(f"File Content ({file_path}):")
                print(file_content)
        else:
            print(f"File {file_path} not found.")
    
    logging.info('Finished querying Doc2Vec model')
