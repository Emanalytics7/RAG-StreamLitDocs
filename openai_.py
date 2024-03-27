import openai
from openai import OpenAI


class OpenAIModel:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
    def create_embedding(self, text, model='text-embedding-3-large'):
        """
        Create an embedding for the given text and store it in ChromaDB.

        """
        embeddings_response = self.client.embeddings.create(input=[text],
                                              model=model)
        
        return embeddings_response.data[0].embedding

        
    def generate_augmented_text(self, prompt, context='',
                                model='gpt-3.5-turbo', max_tokens=500):
        """
        Generate text based on the original prompt and related content.
        """
        augmented_prompt = f'Respond to the following question: {prompt} only from the below context :{context}.'
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': augmented_prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'Error generating text: {str(e)}'
        

