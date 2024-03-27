from openai import OpenAI
import logging
# from document_processor import DocumentProcessor
# from text_chunker import TextChunker

logging.basicConfig(level=logging.INFO)

class TextEmbedder:
    """
    A class to generate embeddings for given text using the OpenAI API.
    """

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
    def generate_embedding(self, text, model='text-embedding-3-small'):
        try:
            text = text.replace('\n', '')
            response = self.client.embeddings.create(input = [text], model=model).data[0].embedding
            return response
        except Exception as e:
            logging.exception(f'Failed to generate embedding: {e}')
            return []

# processor = DocumentProcessor()
# text_from_pdf = processor._get_pdf_text('sample-pdf-file.pdf')
# chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
# chunks = chunker.chunk_text(text_from_pdf)
# # for chunk in chunks:
# #     print(chunk)

# text_embedder = TextEmbedder(api_key='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')
# embeddings = [text_embedder.generate_embedding(chunk) for chunk in chunks]
# print(embeddings)