from select_index_frontend import *
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriving_ans import *
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import asyncio



@st.cache_resource
def get_bot(index_name):
    try:
        return initialize(index_name)
    
        
    except Exception as e:
        st.error(f"Error initializing bot: {e}")

def clear_cache_session():
    st.session_state.clear()
    st.cache_resource.clear()

async def type_text(container, text, delay=0.05):
    full_text = ""
    for char in text:
        full_text += char
        container.markdown(full_text)
        await asyncio.sleep(delay)
    

def run_chat(ans_retrieval):
    st.write("Chatbot-For everyone")
    
    if 'flow_msg' not in st.session_state:
        st.session_state.flow_msg = [SystemMessage(content="You are AI information-search-explaining and retrieval chatbot"),]
    
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    
    input_text = st.text_input("You: ", "")
    
    if input_text:
        res = run_query(ans_retrieval, input_text)
        st.session_state.flow_msg.append(HumanMessage(content=input_text))
        st.session_state.flow_msg.append(AIMessage(content=res))
        st.session_state.conversation.append((input_text, res))
    
    convo_placeholder = st.container()
   #The loop checks if the current message is the last one in the conversation (i == len(st.session_state['conversation']) - 1).
      #If it is, it applies the typing effect using asyncio.run(type_text(placeholder, f"AI assistant: {ai_msg}")).
#For all other messages, it directly writes the full message.
    with convo_placeholder:
        for i, (user_msg, ai_msg) in enumerate(st.session_state['conversation']):
            with st.chat_message("user"):
                st.write(f"You: {user_msg}")

            if ai_msg is not None:
                with st.chat_message("assistant"):
                    if i==len(st.session_state['conversation'])-1:
                        placeholder=st.empty()
                        asyncio.run(type_text(placeholder, f"AI assistant: {ai_msg}"))
                    else:
                        st.write(f"AI assistant: {ai_msg}")
                
            

def main():
    with st.sidebar:
        sel_index()
    if st.button('RELOAD'):
        clear_cache_session()
        st.rerun()
    if 'sel_index' in st.session_state and st.session_state['index_taken']:
        ans_retrieval = get_bot(st.session_state['sel_index'])
        if ans_retrieval:
            run_chat(ans_retrieval)

if __name__ == "__main__":
    main()
