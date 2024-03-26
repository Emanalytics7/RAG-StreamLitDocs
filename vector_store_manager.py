import chromadb
import logging
import os

class ChromaDBStorage:
    def __init__(self, db_path='./chroma_storage', collection_name='embedding6'):
        self.db_path = db_path
        self.collection_name = collection_name
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(db_path)
        self.collection = self.client.create_collection(collection_name)


    # def _get_or_create_collection(self, name):
    #     existing_collections = self.client.list_collections()
    #     if name not in existing_collections:
    #         self.client.create_collection(name=name)

    #     return self.client.get_collection(name=name)           

    def store_embedding(self, id, embedding):
        self.collection.add(ids=[id], embeddings=[embedding])

    def query_embedding(self, embedding, n_results=5):
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
