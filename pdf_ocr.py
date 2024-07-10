import os
import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError
import pytesseract
from pymupdf import FileDataError
from tqdm import tqdm

# 定义扫描文件夹的函数
def scan_folder(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

# 定义进行OCR识别的函数
def ocr_pdf(pdf_file):
    try:
        # 使用PyMuPDF打开PDF文件
        doc = fitz.open(pdf_file)
        text = ''

        # 遍历每一页，提取文本
        for page_num in tqdm(range(len(doc)), desc=f'Processing {os.path.basename(pdf_file)}', unit='page'):
            page = doc.load_page(page_num)
            text += page.get_text()

        # 如果文本为空，尝试进行OCR识别
        if not text:
            # 使用Pillow打开PDF文件，并转换为图像
            images = []
            pdf_image = Image.open(pdf_file)
            images.append(pdf_image.convert('RGB'))

            # 使用pytesseract进行OCR识别
            for image in images:
                text += pytesseract.image_to_string(image, lang='eng')

        return text
    except (FileDataError, UnidentifiedImageError) as e:
        print(f"Failed to process {pdf_file}: {str(e)}")
        return None

# 定义保存文本到文件的函数（保存到PDF同级目录下）
def save_text_to_file(text, pdf_file):
    pdf_path, pdf_filename = os.path.split(pdf_file)
    txt_filename_base = os.path.splitext(pdf_filename)[0]
    max_chars_per_file = 4080

    for i in range(0, len(text), max_chars_per_file):
        part_num = i // max_chars_per_file + 1
        txt_filename = f"{txt_filename_base}_part{part_num}.txt"
        txt_path = os.path.join(pdf_path, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as file:
            file.write(text[i:i + max_chars_per_file])
        print(f'Saved part {part_num} as {txt_filename}')

if __name__ == "__main__":
    # 设置扫描的文件夹路径
    folder_path = r'C:\Users\skyve\Documents\GitHub\AI-powered-files-system-and-the-searching-framwork\pdfb'
    
    # 扫描文件夹下的所有PDF文件
    pdf_files = scan_folder(folder_path)
    
    # 对每个PDF文件进行OCR扫描，并保存到文本文件
    for pdf_file in tqdm(pdf_files, desc='Processing all PDFs', unit='file'):
        text = ocr_pdf(pdf_file)  # 进行OCR扫描
        if text is not None:
            save_text_to_file(text, pdf_file)  # 保存文本到文件
            print(f'{pdf_file} processed and saved as parts.')
        else:
            print(f'{pdf_file} skipped due to processing error')
