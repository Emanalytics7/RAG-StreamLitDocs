import chromadb
import os
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from text_embedder import TextEmbedder


class ChromaDBStorage:
    def __init__(self, db_path='./chroma_db', collection_name='unique'):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(db_path)
        self.collection = self.client.create_collection(collection_name)


    def get_or_create_collection(self, name):
        try:
            existing_collections = self.client.list_collections()
            if name not in existing_collections:
                collection = self.client.create_collection(name=name)
            else:
                collection = self.client.get_collection(name)
            return collection
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def store_embedding(self, id, embedding, metadata, text):
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
            ids=[id]
        )

    def query_embedding(self, embedding, n_results=5):
        query_results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        return [result['id'] for result in query_results]
    
    def retrieve_text(self, id):
        result = self.collection.get(id=id)
        if result:
            return result['documents']
        else:
            return None
# chroma_db = ChromaDBStorage(collection_name='chroma_uniqu3')
# processor = DocumentProcessor()
# text_from_pdf = processor._get_pdf_text('sample-pdf-file.pdf')
# chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
# chunks = chunker.chunk_text(text_from_pdf)
# # for chunk in chunks:
# #     print(chunk)
# text_embedder = TextEmbedder(api_key='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')
# embeddings = [text_embedder.generate_embedding(chunk) for chunk in chunks]
# for i, embedding in enumerate(embeddings):
#     embedding_id = f'chunk_{i}'
#     chroma_db.store_embedding(id=embedding_id, embedding=embedding)