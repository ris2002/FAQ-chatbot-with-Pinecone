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
import requests
from ratelimit import limits, sleep_and_retry

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
        # Retry logic with exponential backoff
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                # Generate a response using the retrieval chain
                time.sleep(60)
                response = retrieval_chain.invoke(
                    {"input": input_text, "chat_history": st.session_state.flow_msg},
                    config={"configurable": {"session_id": f'{session_id}'}}
                )
                return response['answer']
            except requests.exceptions.RateLimitError as e:
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        else:
            st.error("Failed to retrieve response after multiple attempts.")
            return None
    except KeyError as e:
        st.error(f"KeyError occurred: {e}. Check the response structure.")
        return None

def ini_embed():
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    return embeddings

def ini_prompt():
    prompt = ChatPromptTemplate.from_template('''
       The {context} consists of course curriculm of UNDERGRADUATE PROGRAMME B.Tech. of NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL
SRINIVASNAGAR PO, MANGALORE 575 025 KARNATAKA, INDIA.Curriculm means it consists of Regulations (General) of the college ,Regulations of  UG,
Forms & Formats of UG, Course Structure of UG and Course Contents of  UG. Your job is to guide the student based on his/her intersted course and make their decision process easier.
                                              The user only asks regading the {context}.
                                              It is important to give the contact details of clgis he or she feels the need to contact the the clg.




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
        lambda session_id: get_session_history(session_id),
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
