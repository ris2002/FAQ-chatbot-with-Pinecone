import streamlit as st
from pinecone import Pinecone as Pineconex, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['PINECONE_API_KEY'] = '81401caf-7ceb-4cf2-b38f-4f57374b8ec8'  # Replace with your actual Pinecone API key
key_az="6ba3839d3bcd40beb8d81cf3d9255677"
end="https://rishil123.cognitiveservices.azure.com/"
# Initialize Pinecone client
pc = Pineconex(api_key=os.getenv('81401caf-7ceb-4cf2-b38f-4f57374b8ec8'))
analysis_features = ["ocrHighResolution"]
def extract_embeddings_upload_index(pdf_path, index_name):
    print(f"Loading PDF from path: {pdf_path}")
    
    # Load PDF documents
    txt_docs = PyPDFium2Loader(pdf_path).load()
    #txt_docs=AzureAIDocumentIntelligenceLoader(api_key="6ba3839d3bcd40beb8d81cf3d9255677",file_path=pdf_path,api_endpoint="https://rishil123.cognitiveservices.azure.com/",api_model="prebuilt-layout").load()
    

    # Split documents
    print("Splitting documents...")
    splt_docs = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = splt_docs.split_documents(txt_docs)
    print(f"Split into {len(docs)} chunks")

    # Initialize OpenAI embeddings
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

    # Upload documents to Pinecone index
    print("Initializing Pinecone Vector Store...")
    dbx = PineconeVectorStore.from_documents(documents=docs, index_name=index_name, embedding=embeddings)
    print(f"Uploaded {len(docs)} documents to Pinecone index '{index_name}'")

def create_index():
    if 'details_taken' not in st.session_state:
        st.session_state['details_taken'] = False
    
    if 'new_index' not in st.session_state:
        name = st.text_input("Enter your index name:")
        dimensions = 1536  # Assuming default dimensions for embedding model
        st.write('Dimensions set to 1536 since embedding model used="text-embedding-ada-002"')
        metrics = st.text_input("Enter your metric (cosine, euclidean, dotproduct):")
        
        if st.button("Create Index", key="create_index_button"):
            if name and metrics:
                st.session_state['new_index'] = {
                    'name': name,
                    'dimension': dimensions,
                    'metric': metrics
                }
                st.write(f"Index '{name}' details have been stored.")
            else:
                st.warning("Please provide both index name and metric.")

def sel_index():
    create_index()
    if 'index_name' not in st.session_state:
        st.session_state['index_name']=None
    if 'index_taken' not in st.session_state:
        st.session_state['index_taken']=False



    if 'new_index' in st.session_state:
        index_details = st.session_state['new_index']
        try:
            pc.create_index(name=index_details['name'], dimension=index_details['dimension'], metric=index_details['metric'], spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            st.write(f"Index '{index_details['name']}' has been created")
        except Exception as e:
            st.error(f"Error creating index: {str(e)}")

    indexes = pc.list_indexes().names()
    sel_index = st.selectbox("Select an index", indexes)

    if st.button("CONFIRM"):
        st.session_state['index_name'] = sel_index
        st.session_state['index_taken']=True
        st.write(f"Selected index: {sel_index}")

def sel_mod():
    with st.sidebar:
        sel_index()

if __name__ == "__main__":
    sel_mod()
