import os
import logging
from docx import Document
import PyPDF2

logging.basicConfig(level=logging.INFO)

class DocumentProcessor:
    @staticmethod
    def get_text(uploaded_file):
        if not uploaded_file or uploaded_file.size == 0:
            logging.error('Empty or None file provided.')
            return 'File is empty or not provided!'
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.pdf':
            return DocumentProcessor._get_pdf_text(uploaded_file)
        elif file_extension == '.docx':
            return DocumentProcessor._get_docx_text(uploaded_file)
        else:
            logging.error(f'Unsupported file type: {file_extension}')
            return 'Unsupported File Type!'
        
    @staticmethod
    def _get_pdf_text(pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text
        except Exception as e:
            logging.exception(f'Error reading PDF: {e}')
            return f'Error reading PDF: {str(e)}'
   
    @staticmethod
    def _get_docx_text(docx_file):
        try:
            doc = Document(docx_file)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logging.exception(f'Error reading DOCX: {e}')
            return f'Error reading DOCX: {str(e)}'


# Example for Testing Document Processor
# processor = DocumentProcessor()
# text_from_pdf = processor._get_pdf_text('sample-pdf-file.pdf')
# text_from_docx = processor._get_docx_text('sample-docx-file-for-testing.docx')
# print(text_from_pdf)
# print(text_from_docx)





