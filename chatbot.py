from select_index_frontend import *
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriving_ans import *

@st.cache_resource
def get_bot(index_name):
   try:
      return initialize(index_name)
   except KeyError as e:
      st.error(f"KeyError occurred: {e}. Check the response structure.")
   except Exception as e:
      st.error(f"Error initializing bot: {e}")


def clear_cache_session():
   st.session_state.clear()
   st. cache_resource. clear
def run_chat(ans_retrival):
   st.title("Chatbot-For everyone")
   input_text = st.text_input("You: ", "")
   if input_text:
      res=run_query(ans_retrival,input_text)
      st.text_area("AI assistant:", value=res, height=500)

      
      


def main():
    with st.sidebar:
       sel_index()
    if st.button('RELOAD'):
       clear_cache_session()
       st.experimental_rerun()
    if 'sel_index' in st.session_state and st.session_state['index_taken']:
       ans_retrieval = get_bot(st.session_state['sel_index'])
       if ans_retrieval:
          run_chat(ans_retrieval)
          
          


if __name__ == "__main__":
 main()
    