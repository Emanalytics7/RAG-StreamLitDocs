import os
import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from vector_store_manager import ChromaDBStorage
from text_embedder import TextEmbedder
from document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

class QueryHandler:
    def __init__(self, api_key, model_name='gpt-3.5-turbo'):
        self.api_key = api_key
        self.model_name = model_name
        self.parser = StrOutputParser()
        self.model = ChatOpenAI(openai_api_key=self.api_key, model=self.model_name)
        self.setup_chains()

    def setup_chains(self):
        # Setup the primary response chain
        template = """
        Answer the question based on the context below. If you can't
        answer the question, reply "I'm so sorry! But I don't have any context related to this query! :)"
        Context: {context}
        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.model | self.parser

        # Setup the translation chain
        translation_template = 'Translate {answer} to {language}'
        self.translation_prompt = ChatPromptTemplate.from_template(translation_template)
        self.translation_chain = (
            {'answer': self.chain, 'language': itemgetter('language')} |
            self.translation_prompt |
            self.model |
            self.parser
        )

    def handle_query(self, context, question, language='English'):
        response = self.translation_chain.invoke(
            {
                'context': context,
                'question': question,
                'language': language
            }
        )
        return response
# processor = DocumentProcessor()
# pdf_text = processor._get_pdf_text('sample-pdf-file.pdf')
# user_query = 'What is the main theme here?'
# text_embedder = TextEmbedder()
# query_embedding = text_embedder.generate_embedding(user_query)
# response = QueryHandler.handle_query(pdf_text, user_query)
