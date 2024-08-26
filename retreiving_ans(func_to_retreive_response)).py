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
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationChain

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

open_api_key = os.getenv('OPENAI_API_KEY')
langchain_key = os.getenv('LANGCHAIN_API_KEY')
//all keys are wrong
os.environ['PINECONE_API_KEY'] = '81401caf-7ceb-4cf2-b38fdgfsg-4f57374b8ec8'
os.environ['OPENAI_API_KEY'] = "sk-br-infotech-ddO9Biqt05Yfgf8wc1sKlf7T3BlbkFJ5v5dtLdqf5H65NNDpqxl"
os.environ['LANGCHAIN_API_KEY'] = "lsv2_pt_ed77deb976784gfgf563657868ef4771b979d113f_9d5f4fd170"
os.environ["LANGCHAIN_TRACING_V2"] = "true"



def run_query(retrieval_chain, input_text):
    st.write('run query')
    st.chat_message("human").write(input_text)
    
    try:
        # Retrieve the session key from session state
        keys = st.session_state['session_key']
        
        # Initialize StreamlitChatMessageHistory only once
        
            
        
        mgs=StreamlitChatMessageHistory(key=keys)
        
        
        # Prepare input with history
        #input_with_history = [{"role": msg.type, "content": msg.content} for msg in mgs.] + [{"role": "user", "content": input_text}]
        
        # Generate a response using the retrieval chain
        config = {"configurable": {"session_id": keys}}
        print('fddsgsdg')
        
        response = retrieval_chain.invoke({'input':input_text}, config=config)
        print('sfgfgfdsgvd')
        
        # Display AI response
        st.chat_message("ai").write(response['answer'])
        
        # Add user and AI messages to history
        mgs.add_user_message(input_text)
        mgs.add_ai_message(response['answer'])

        return response['answer']
    except KeyError as e:
        st.error(f"KeyError occurred: {e}. Check the response structure.")
        return None


def ini_embed():
      embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
      return embeddings
def ini_prompt():
     
        system_msg=SystemMessagePromptTemplate.from_template(
        "You are an AI assistant with expertise in {context}, specifically focusing on the provided manual. "
        "Carefully study the manual, particularly chapters or sections related to {context}. "
        "Your responses should be concise, precise, and directly related to {context}. "
        "The user will ask questions related to {context} and its topics only. "
        "If a question is only partially related, clarify the ambiguity and focus on the most relevant aspects. "
        "If the question is unrelated to {context}, politely indicate that it's outside the scope of the manual. "
        "If the user asks what the context is, provide a brief summary of the {context}." )
        human_msg=HumanMessagePromptTemplate.from_template("Question: {input}")
        prompt=ChatPromptTemplate.from_messages([system_msg,MessagesPlaceholder(variable_name="history"),human_msg])
        return prompt 


          
     
def get_session_history(key: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = ChatMessageHistory()
    return st.session_state["chat_history"]
     


def initialize(index_name):
    
        
        st.write('model')

        embeddings = ini_embed()
        st.write('model1')
        dbx = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        st.write('model2')
        llm = ChatOpenAI(model='gpt-4o', temperature=0.5,max_tokens=3000)
        st.write('model3')
        keys=st.session_state['session_key']
        mgs=StreamlitChatMessageHistory(key=keys)
        if len(mgs.messages) == 0:

            mgs.add_ai_message("How can I help you?")

        prompt = ini_prompt()
        st.write('model4')
        doc_chain = create_stuff_documents_chain(llm, prompt)
        #conv_chain=ConversationChain(llm=llm,prompt=prompt,memory=ConversationBufferMemory())
        memory=ConversationBufferMemory(memory_key='history',return_messages=True)
        st.write('model5')
        retriever = dbx.as_retriever()
        st.write('model6')
        run= create_retrieval_chain(retriever, doc_chain)
        st.write('model7')
        ans_retrival=RunnableWithMessageHistory(run,get_session_history,input_messages_key="input",history_messages_key="history",output_messages_key="answer")
        
        
        
        st.write('model8')
        
              
        

        return ans_retrival
    
        
