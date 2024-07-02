from pinecone import *
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from pinecone import Pinecone as Pineconex, ServerlessSpec

os.environ['PINECONE_API_KEY'] = 'PINECONE_API_KEY'

pc = Pineconex(api_key=os.getenv('PINECONE_API_KEY'))





def sel_index():
    
   

    
    if 'sel_index' not in st.session_state:
        st.session_state['sel_index']=None
    if 'index_taken'not in st.session_state:
        st.session_state['index_taken']=False
        

    indexes = pc.list_indexes().names()
    sel_index = st.selectbox("Select an index", indexes)

    if st.button("CONFIRM"):
        st.session_state['sel_index'] = sel_index
        st.write(f"Selected index: {sel_index}")
        st.session_state['index_taken']=True

def sel_mod():
    with st.sidebar:
        sel_index()

if __name__ == "__main__":
    sel_mod()
