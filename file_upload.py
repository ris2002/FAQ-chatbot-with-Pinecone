import streamlit as st
from select_index import *  # Assuming this function exists in select_index.py
import tempfile

def main():
    # Initialize session state variables if not already initialized
    if 'index_taken' not in st.session_state:
        st.session_state['index_taken'] = False
    
    if 'index_name' not in st.session_state:
        st.session_state['index_name'] = None
    
    if 'uploaded_file_info' not in st.session_state:
        st.session_state['uploaded_file_info'] = None

    with st.sidebar:
        # Load index selection widget
        
        sel_index()
        print('mcdmcv')

    # Check if an index has been selected and file upload is allowed
    if st.session_state['index_taken']:
        print('mcdmcvdhqwbdhbuqhwbduhqwb')
        file_uploaded = st.file_uploader("Upload your file", type=['pdf', 'txt', 'csv'], key="unique_file_uploader_key")
        if file_uploaded:
            # Save uploaded file information
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_uploaded.name.split('.')[-1]}") as temp_file:
                temp_file.write(file_uploaded.read())
                temp_file_path = temp_file.name
            
            st.session_state['uploaded_file_info'] = {
                'name': file_uploaded.name,
                'type': file_uploaded.type,
                'temp_file_path': temp_file_path
            }
            st.write(f"Successfully uploaded {file_uploaded.name} to {temp_file_path}")

    # Display uploaded file info if available
    if st.session_state['uploaded_file_info']:
        file_info = st.session_state['uploaded_file_info']
        st.write(f"Successfully uploaded {file_info['name']} to {file_info['temp_file_path']}")

    # Check if both index name and file path are available to proceed with upload
    if 'index_name' in st.session_state and 'uploaded_file_info' in st.session_state and st.session_state['uploaded_file_info']:
        if st.button('UPLOAD'):
            index_name = st.session_state['index_name']
            file_path = st.session_state['uploaded_file_info']['temp_file_path']
            st.write(f"Index Name: {index_name}")
            st.write(f"File Path: {file_path}")
            try:
                extract_embeddings_upload_index(file_path, index_name)
                st.write(f"Successfully uploaded the file to index {index_name}")
            except Exception as e:
                st.error(f"Error uploading file to index {index_name}: {str(e)}")

if __name__ == "__main__":
    main()
