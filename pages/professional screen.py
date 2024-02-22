import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
from prompts import templates



jd = st.text_area("Please enter the job description here (If you don't have one, enter keywords, such as PostgreSQL or Python instead): ")

@dataclass
class Message:
    """class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

# 1: Job Description Vector Embeddings:

def save_vector(text):  #The function save_vector(text) is called to generate vector embeddings for the job description text (jd).
    """embeddings"""
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
     # Create emebeddings
    embeddings = OpenAIEmbeddings(openai_api_key="openaikey")
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch  



def initialize_session_state_jd():
    """ initialize session states """

    # 2: RetrievalQA Chain Setup:
    if 'jd_docsearch' not in st.session_state:
        st.session_state.jd_docserch = save_vector(jd) 
    #The resulting vector embeddings are stored in the session state as jd_docsearch.

    if 'jd_retriever' not in st.session_state: 
        st.session_state.jd_retriever = st.session_state.jd_docserch.as_retriever(search_type="similarity")
    #The jd_docsearch is used to create a retriever for the RetrievalQA chain (jd_retriever).

    if 'jd_chain_type_kwargs' not in st.session_state: #A prompt template specific to job description interviews (Interview_Prompt) is created.
        Interview_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.jd_template)
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_Prompt}  
    #The prompt template is stored in the session state as jd_chain_type_kwargs.
    
    if 'jd_memory' not in st.session_state: 
        st.session_state.jd_memory = ConversationBufferMemory()
    #An instance of ConversationBufferMemory is created and stored in the session state as jd_memory.
    
    # interview history
    if "jd_history" not in st.session_state:
        st.session_state.jd_history = []
        st.session_state.jd_history.append(Message("ai",
                                                   "Hello, Welcome to the interview. I am your interviewer today. I will ask you professional questions regarding the job description you submitted."
                                                   "Please start by introducting a little bit about yourself. Note: The maximum length of your answer is 4097 tokens!"))
    #If the interview history (jd_history) does not exist in the session state, 
    #it is initialized as an empty list. A welcome message from the AI interviewer is added to the history.


    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    
    

    #The guideline for the interview is generated using the RetrievalQA chain, 
    #utilizing the prompt template and job description retriever.
    #The generated guideline is stored in the session state as jd_guideline.
    if "jd_guideline" not in st.session_state:
        llm = ChatOpenAI(
        openai_api_key="openaikey",
        model_name = "gpt-3.5-turbo",
        temperature = 0.8,)

        st.session_state.jd_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.jd_chain_type_kwargs, 
            chain_type='stuff',
            retriever=st.session_state.jd_retriever, 
            memory = st.session_state.jd_memory).run("Create an interview guideline and prepare only one questions for each topic. \
                                                     Make sure the questions tests the technical knowledge")
    
    
    # llm chain and memory
    # Conversation Chain Initialization:
    # An instance of the conversation chain (jd_screen) is created with a prompt template specific to conversation-based interviews.
    # The conversation chain is initialized with the Language Model (LLM) and memory buffer.
    if "jd_screen" not in st.session_state:
        llm = ChatOpenAI(
            openai_api_key="openaikey",
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""I want you to act as an interviewer strictly following the guideline in the current conversation.
                            Candidate has no idea what the guideline is.
                            Ask me questions and wait for my answers. Do not write explanations.
                            Ask question like a real person, only one question at a time.
                            Do not ask the same question.
                            Do not repeat the question.
                            Do ask follow-up questions if necessary. 
                            You name is GPTInterviewer.
                            I want you to only reply as an interviewer.
                            Do not write all the conversation at once.
                            If there is an error, point it out.

                            Current Conversation:
                            {history}

                            Candidate: {input}
                            AI: """)

        st.session_state.jd_screen = ConversationChain(prompt=PROMPT, llm=llm,
                                                           memory=st.session_state.jd_memory)
   
   
    if 'jd_feedback' not in st.session_state:
        llm = ChatOpenAI(
            openai_api_key="openaikey",
            model_name="gpt-3.5-turbo",
            temperature=0.8, )
        
        st.session_state.jd_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.jd_memory,
        )


def answer_call_back():
    with get_openai_callback() as cb:   #The get_openai_callback() function is called to obtain a callback object (cb) for interacting with the OpenAI API. 
                                        #This allows the function to access the OpenAI language model for generating responses.
        # user input
        human_answer = st.session_state.answer

        input = human_answer

        st.session_state.jd_history.append(
            Message("human", input)
        )    #The user's input (either typed or transcribed) is appended to the interview history (st.session_state.jd_history)
             #with the origin set to "human", indicating that it came from the user.
        
        # OpenAI answer and save to history
        llm_answer = st.session_state.jd_screen.run(input)   #The conversation chain (jd_screen) is used to generate an AI response (llm_answer) based on the user's input.
    
        st.session_state.jd_history.append(
            Message("ai", llm_answer)
        )  #The AI response is appended to the interview history with the origin set to "ai", indicating that it came from the AI interviewer.

        st.session_state.token_count += cb.total_tokens #The token count (st.session_state.token_count) is updated by adding the total tokens consumed by the OpenAI callback (cb.total_tokens).

        return llm_answer



if jd:
    # initialize session states
    initialize_session_state_jd()

    #st.write(st.session_state.jd_guideline)
    
    credit_card_placeholder = st.empty() #The progress of the interview is displayed as a caption

    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")

    chat_placeholder = st.container() #The chat history, including messages exchanged between the user and the AI interviewer
    answer_placeholder = st.container()


    # if submit email adress, get interview feedback imediately
    if guideline:
        st.write(st.session_state.jd_guideline)
    if feedback:
        evaluation = st.session_state.jd_feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            
            answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                answer = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.jd_history:
                if answer.origin == 'ai':
                    with st.chat_message("assistant"):
                        st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
        Progress: {int(len(st.session_state.jd_history) / 30 * 100)}% completed.""")
else:
    st.info("Please submit a job description to start the interview.")