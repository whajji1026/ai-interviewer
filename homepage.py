import streamlit as st
from app_utils import switch_page
from PIL import Image

st.set_page_config(page_title="AI Interviewer", layout="centered")

home_title = "AI Interviewer"
home_introduction = "Welcome to AI Interviewer, empowering your interview preparation with generative AI."
with st.sidebar:
    st.markdown('AI Interviewer Project')
    
st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""", unsafe_allow_html=True)
st.markdown("""\n""")
st.info("""
    📚In this session, the AI Interviewer will assess your technical skills as they relate to the job description.
    Note: The maximum length of your answer is 4097 tokens!
    - Each interview will take 10 to 15 minutes.
    - To start a new session, simply refresh the page.
    - Start by introducing yourself and enjoy! """)
if st.button("Start Interview!"):
        switch_page("Professional Screen")

