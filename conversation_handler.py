from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import openai

class ConversationChainManager:
    def __init__(self, llm, retriever, openai_api_key, memory_key='chat_history'):
        self.llm = llm
        self.retriever = retriever
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

    def get_conversation_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.as_retriever(),
            memory=self.memory
        )
    
    def query_gpt3_5_turbo(self, question, context=''):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=f"{context}\n\nQuestion: {question}\nAnswer:",
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=["\n"],
        )
        return response.choices[0].text.strip()
