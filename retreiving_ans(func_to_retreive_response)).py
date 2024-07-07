import os
import uuid
import time
import streamlit as st
from pinecone import *
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEndpoint
import asyncio
from langchain.chains import ConversationChain

# Set environment variables
os.environ['PINECONE_API_KEY'] = '81401caf-7ceb-4cf2-b38f-4f57374b8ec8'
os.environ['OPENAI_API_KEY'] = "sk-br-infotech-ddO9Biqt05Y8wc1sKlf7T3BlbkFJ5v5dtLdqf5H65NNDpqxl"
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

os.environ["hugging_face_repo"]="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"

session_id = str(uuid.uuid4())



     

def run_query(retrieval_chain, input_text):
    st.write('run query')
    try:
        # Generate a response using the retrieval chain
    
        response =  retrieval_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": f'{session_id}'}}
        )
        
        return response['answer']
    except KeyError as e:
        st.error(f"KeyError occurred: {e}. Check the response structure.")
        return None

def ini_embed():
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    return embeddings

def ini_prompt():
    prompt = ChatPromptTemplate.from_template('''
        You are an AI assistant with expertise in {context}, specifically focusing on the provided manual.
        What you should do is study the manual carefully and answer to the questions accordingly.
        The user will ask only questions related to the {context} and its topics.
        If you feel the question asked is unrelated to {context} or its topics do not answer it.
        If the user asks what the context is, just give the summary of the {context}.

        <context>{context}</context>  
        Question: {input}
    ''')
    return prompt


def get_session_history(session_id: str) -> BaseChatMessageHistory:
        print('Session_chat')
        
        print(StreamlitChatMessageHistory(key=session_id))
        
        return StreamlitChatMessageHistory(key=session_id)

def initialize(index_name):
    embeddings = ini_embed()
    print('11')
    dbx = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    print('12')
    llm = ChatOpenAI(model='gpt-4o', temperature=0.5, max_tokens=3000)
    
   # model_id="meta-llama/Meta-Llama-3-8B"
   #model=AutoModelForCausalLM.from_pretrained(model_id)
    #tokenizer=AutoTokenizer.from_pretrained(model)
    #pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=5000)
    #repo_id="meta-llama/Llama-2-7b-hf"

    print('13')
    prompt = ini_prompt()
    print('14')
    doc_chain = create_stuff_documents_chain(llm, prompt)
    print('15')
    retriever = dbx.as_retriever()
    print('16')
    ans_retrieval = create_retrieval_chain(retriever, doc_chain)
    print('17')

    

        
    
    # Wrap the retrieval chain with RunnableWithMessageHistory
    conversational_ans_retrieval = RunnableWithMessageHistory(
        ans_retrieval,
        lambda session_id: StreamlitChatMessageHistory(key=session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
        
    )
    print('17')
    
    print(session_id)
    print('18')
    

    print('conversational_ans_retrieval***************************************************************************************************')
    print(get_session_history(session_id))
    
    

    return conversational_ans_retrieval
