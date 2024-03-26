from openai import OpenAI
import openai
class OpenAIModel:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    def create_embedding(self, text, model='text-embedding-3-large'):
        """
        Create an embedding for the given text and store it in ChromaDB.
        """
        try:
            embeddings = self.client.embeddings.create(input=[text],
                                                        model=model).data[0].embedding
            return embeddings
        except Exception as e:
            return f'Error creating or storing embedding: {str(e)}'

    def generate_augmented_text(self, prompt, context='',
                                model='gpt-3.5-turbo', max_tokens=100):
        """
        Generate text based on the original prompt and related content.
        """
        augmented_prompt = f'{context}\n\n{prompt}'
        try:
            response = openai.completions.create(
                model=model,
                messages=[{"role": "system", "content": augmented_prompt}],
                temperature=0,
                max_tokens=500
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f'Error generating text: {str(e)}'
