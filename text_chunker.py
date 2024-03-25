from langchain.text_splitter import CharacterTextSplitter
from document_processor import DocumentProcessor

class TextChunker:
    def __init__(self, separator='\n', chunk_size=1000, chunk_overlap=20):
        self.text_splitter = CharacterTextSplitter(separator=separator,
                                                    chunk_size=chunk_size,
                                                     chunk_overlap=chunk_overlap,
                                                      length_function=len
                                                   )
        
    def chunk_text(self, document):
        return self.text_splitter.create_documents([document])


# Example for Testing TextChunker
# processor = DocumentProcessor()
# text_from_pdf = processor.get_pdf_text('sample-pdf-file.pdf')
# chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
# chunks = chunker.chunk_text(text_from_pdf)[0]
# print(chunks)