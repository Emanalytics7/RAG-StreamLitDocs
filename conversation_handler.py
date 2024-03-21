from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class ConversationChainManager:
    def __init__(self, llm, retriever, memory_key='chat_history'):
        self.llm = llm
        self.retriever = retriever
        self.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

    def get_conversation_chain(self):
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.as_retriever(),
            memory=self.memory
        )
