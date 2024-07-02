import os
import csv
import logging
from tqdm import tqdm

# Setup logging
log_directory = './log'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename='./log/log.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logging.info('Start processing text files')

MAX_FILE_SIZE_BYTES = 131072  # 128 KB

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

def remove_newlines(text):
    """Remove all newline characters from text."""
    return text.replace('\n', ' ').replace('\r', '')

def write_files_to_csv(txt_files, output_csv):
    """Write file paths, file number, and text content to a CSV file."""
    logging.info(f'Writing {len(txt_files)} text files to CSV')
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'File Number', 'File Text'])
        for i, txt_file in enumerate(tqdm(txt_files, desc="Processing text files", unit="file")):
            try:
                file_size = os.path.getsize(txt_file)
                if file_size > MAX_FILE_SIZE_BYTES:
                    logging.warning(f'Skipping file {txt_file} due to size limit ({file_size} bytes)')
                    continue
                with open(txt_file, 'r', encoding='utf-8') as txt:
                    file_text = remove_newlines(txt.read())
                    writer.writerow([txt_file, i + 1, file_text])
            except Exception as e:
                logging.error(f'Error processing file {txt_file}: {e}')
    logging.info(f'Finished writing text files to CSV')

def process_text_files(input_directory, output_csv):
    """Process all text files in a directory and save file paths, numbers, and text to a CSV file."""
    txt_files = read_txt_files_recursive(input_directory)
    
    write_files_to_csv(txt_files, output_csv)
    
    logging.info(f'Processed all text files. File paths, numbers, and text are saved in "{output_csv}".')
    print(f"Processed all text files. File paths, numbers, and text are saved in '{output_csv}'.")

if __name__ == '__main__':
    input_directory = './pdfs'  # Update with your directory path
    output_csv = './text_files.csv'
    
    process_text_files(input_directory, output_csv)
