from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o-mini", 
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)