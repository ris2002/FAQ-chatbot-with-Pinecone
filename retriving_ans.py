from pinecone import *
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from pinecone import Pinecone as Pineconex, ServerlessSpec
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

open_api_key = os.getenv('OPENAI_API_KEY')
langchain_key = os.getenv('LANGCHAIN_API_KEY')
os.environ['PINECONE_API_KEY'] = '81401caf-7ceb-4cf2-b38f-4f57374b8ec8'
os.environ['OPENAI_API_KEY'] = "sk-br-infotech-ddO9Biqt05Y8wc1sKlf7T3BlbkFJ5v5dtLdqf5H65NNDpqxl"
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"



def run_query(retrieval_chain, input_text):
    st.write('run query')
    try:
        # Generate a response using the retrieval chain
        response = retrieval_chain.invoke({'input': input_text})
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
            If the user asks the the the concontext is ,just give the summary of the {context}.

            <context>{context}</context>  
            Question: {input}
        ''')
     return prompt
     
     


def initialize(index_name):
    
        
        st.write('model')

        embeddings = ini_embed()
        st.write('model1')
        dbx = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        st.write('model2')
        llm = ChatOpenAI(model='gpt-4o', temperature=0.5,max_tokens=3000)
        st.write('model3')
        prompt = ini_prompt()
        st.write('model4')
        doc_chain = create_stuff_documents_chain(llm, prompt)
        st.write('model5')
        retriever = dbx.as_retriever()
        st.write('model6')
        ans_retrival = create_retrieval_chain(retriever, doc_chain)
        st.write('model7')

        return ans_retrival
    
        

