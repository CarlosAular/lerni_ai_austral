import os
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from modules.prompt import system_prompt
from dotenv import load_dotenv
load_dotenv()

os.getenv("OPENAI_API_KEY")

openaiModels = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-0125"]



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5, model=openaiModels[0], max_tokens=4000)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": system_prompt}
    )
    return conversation_chain


