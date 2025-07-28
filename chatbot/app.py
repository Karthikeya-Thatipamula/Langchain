from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set API key for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide responses to user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("Langchain Demo With Groq API")
input_text = st.text_input("Search the topic you want")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
output_parser = StrOutputParser()

# Create chain: Prompt → LLM → Output parser
chain = prompt | llm | output_parser

# Execute chain when user provides input
if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)