# import openai
# from openai import OpenAI


# class OpenAIModel:
#     def __init__(self, api_key):
#         self.client = OpenAI(api_key=api_key)
#         self.api_key = api_key
#     def create_embedding(self, text, model='text-embedding-3-large'):
#         """
#         Create an embedding for the given text and store it in ChromaDB.

#         """
#         embeddings_response = self.client.embeddings.create(input=[text],
#                                               model=model)
        
#         return embeddings_response.data[0].embedding

        
#     def generate_augmented_text(self, prompt, context='',
#                                 model='gpt-3.5-turbo', max_tokens=500):
#         """
#         Generate text based on the original prompt and related content.
#         """
#         augmented_prompt = f'Respond to the following question: {prompt} only from the below context :{context}.'
#         try:
#             response = openai.chat.completions.create(
#                 model=model,
#                 messages=[{'role': 'system', 'content': augmented_prompt}],
#                 temperature=0,
#                 max_tokens=max_tokens
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f'Error generating text: {str(e)}'
import openai
from vector_store_manager import ChromaDBStorage  # Assuming you have a class like this

class QueryHandler:
    def __init__(self, db_path, collection_name, openai_api_key):
        """
        Initialize the query handler with a database storage and OpenAI API key.
        """
        self.db_storage = ChromaDBStorage(db_path, collection_name)
        openai.api_key = openai_api_key

    def generate_embedding(self, text):
        """
        Generate an embedding for the given text using OpenAI's API.
        """
        response = openai.embeddings.create(input=text, model="text-embedding-3-small")
        embedding = response.data[0].embedding
        return embedding

    def create_and_store_embeddings(self, text_chunks):
        """
        For each text chunk, generate an embedding and store it using db_storage.
        Assumes text_chunks is a list of tuples [(id, text), ...].
        """
        for text_id, text in text_chunks:
            embedding = self.generate_embedding(text)
            self.db_storage.store_embedding(text_id, embedding)

    def answer_query(self, query):
        """
        Answer a query based on the document's content by finding the most relevant text chunk.
        """
        query_embedding = self.generate_embedding(query)
        query_results = self.db_storage.query_embedding(query_embedding, n_results=1)
        
        if query_results:
            most_relevant_id = query_results[0]['id']
            most_relevant_text = self.db_storage.get_text_by_id(most_relevant_id)
            
            response = openai.chat.completions.create(
                prompt=f"Given the text: \"{most_relevant_text}\". Answer the question: {query}",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            answer = response.choices[0].text.strip()
            return answer
        else:
            return "Sorry, I couldn't find a relevant answer in the document."

# Example usage
if __name__ == '__main__':
    db_path = './chroma'
    collection_name = 'collecti_nam'
    openai_api_key = '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'
    
    handler = QueryHandler(db_path, collection_name, openai_api_key)
    # Assume you have text chunks ready to be processed
    text_chunks = [('id1', 'Story line'), ('id2', 'Movie')]
    
    handler.create_and_store_embeddings(text_chunks)
    
    query = "What is the main theme of the text?"
    print(handler.answer_query(query))


