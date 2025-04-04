 

import streamlit as st
import json
import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="EvalAI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 EvalAI Chatbot")
st.markdown(
    "### Get insights into the EvalAI codebase! 📚\n"
    "Ask anything about EvalAI, and this chatbot will fetch relevant details for you. 🚀"
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

data_file_path = "data.txt"
data_loaded = os.path.exists(data_file_path)

def clean_text(text):
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def generate_keywords(query):
    """Uses AI to generate relevant search keywords based on the query."""
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = f"Generate multiple relevant search keywords for the following query: {query}"
    data = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    
    response = requests.post(gemini_url, headers=headers, data=data)
    if response.status_code == 200:
        keywords = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", query)
        return keywords.split(",")
    return [query]

def search_in_file(file_path, queries):
    relevant_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if any(query.lower() in line.lower() for query in queries):
                relevant_lines.append(line.strip())
    return " ".join(relevant_lines)

def load_and_chunk_text(text_data, chunk_size=1500, overlap=100):
    clean_data = clean_text(text_data)
    chunks = [clean_data[i:i + chunk_size] for i in range(0, len(clean_data), chunk_size - overlap)]
    return chunks

if api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("💬 Ask about EvalAI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if not data_loaded:
            response = "⚠️ data.txt file not found. Please create this file with your EvalAI data."
        else:
            # Search and load relevant data for this specific question
            with st.spinner("Searching for relevant information..."):
                keywords = generate_keywords(prompt)
                relevant_text = search_in_file(data_file_path, keywords)
                
                if not relevant_text:
                    context = "No specific information found about this query in the EvalAI documentation."
                else:
                    doc_chunks = load_and_chunk_text(relevant_text)
                    context = "\n\n".join(doc_chunks)[:4000]
            
            enhanced_prompt = f"""
            Answer the following question about EvalAI based on this context:
            
            Context: {context}
            
            Question: {prompt}
            
            If the context doesn't contain enough information to answer the question, 
            please respond based on general knowledge about machine learning evaluation platforms.
            Keep your answer focused on EvalAI functionality and features.
            """
            
            gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = json.dumps({"contents": [{"parts": [{"text": enhanced_prompt}]}]})
            gemini_response = requests.post(gemini_url, headers=headers, data=data)
            
            if gemini_response.status_code == 200:
                response = gemini_response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
            else:
                response = f"⚠️ Error: Unable to fetch response from Gemini API. Status code: {gemini_response.status_code}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("⚠️ Please add your Gemini API key to continue.", icon="🔐")

st.markdown("---")
st.markdown("📘 *Powered by EvalAI Knowledgebase & Gemini AI!* 🚀")