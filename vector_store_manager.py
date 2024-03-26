import chromadb
import os
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
    
    def store_embedding(self, id, embedding):
        results = self.collection.add(ids=[id], embeddings=embedding)
        return results 

    def query_embedding(self, embedding, n_results=5):
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
