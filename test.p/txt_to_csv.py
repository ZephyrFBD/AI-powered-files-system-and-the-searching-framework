import os
import csv
import logging

# Setup logging
logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start processing text files')

def read_txt_files_recursive(directory):
    """Read all text files from a directory and its subdirectories."""
    logging.info(f'Reading text files from directory: {directory}')
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_files.append(file_path)
                logging.info(f'Found text file: {file_path}')
    return txt_files

def split_into_paragraphs(text):
    """Split text into paragraphs."""
    paragraphs = text.split('\n\n')
    logging.info(f'Split text into {len(paragraphs)} paragraphs')
    return paragraphs

def write_paragraphs_to_csv(paragraphs, original_path, output_csv):
    """Write paragraphs to a CSV file with original path and paragraph number."""
    logging.info(f'Writing {len(paragraphs)} paragraphs from {original_path} to CSV')
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for i, paragraph in enumerate(paragraphs):
            writer.writerow([original_path, i + 1, paragraph])
    logging.info(f'Finished writing paragraphs from {original_path} to CSV')

def process_text_files(input_directory, output_csv):
    """Process all text files in a directory and save paragraphs to a CSV file."""
    logging.info('Initializing the CSV file with headers')
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Paragraph Number', 'Paragraph Text'])
    
    txt_files = read_txt_files_recursive(input_directory)
    
    for txt_file in txt_files:
        logging.info(f'Processing file: {txt_file}')
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                text = file.read()
                paragraphs = split_into_paragraphs(text)
                write_paragraphs_to_csv(paragraphs, txt_file, output_csv)
        except Exception as e:
            logging.error(f'Error processing file {txt_file}: {e}')

if __name__ == '__main__':
    input_directory = './books'  # Update with your directory path
    output_csv = './paragraphs.csv'
    
    process_text_files(input_directory, output_csv)
    
    logging.info(f'Processed all text files. Paragraphs are saved in "{output_csv}".')
    print(f"Processed all text files. Paragraphs are saved in '{output_csv}'.")
