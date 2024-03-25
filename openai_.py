import openai
class OpenAIModel:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
  

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

    def generate_augmented_text(self, prompt, related_content, 
                                model='gpt-3.5-turbo', max_tokens=100):
        """
        Generate text based on the original prompt and related content.
        """
        augmented_prompt = f"{prompt} {related_content}"  
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=augmented_prompt,
                max_tokens=max_tokens,
                temperature=0.7  
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f'Error generating text: {str(e)}'
