# # # import streamlit as st
# # # import requests
# # # import json
# # # import os
# # # from dotenv import load_dotenv

# # # # Load environment variables
# # # load_dotenv()

# # # # Customizing UI for better user experience
# # # st.set_page_config(page_title="Challenge Support Chatbot", page_icon="💡", layout="centered")

# # # # Show title and description with enhanced UI
# # # st.title("💡 Chat Bot for Enhanced Challenge Support")
# # # st.markdown(
# # #     "### Welcome to your personal support assistant! 🤖💙\n"
# # #     "This chatbot is designed to provide instant support and guidance. "
# # #     "Whether you need help with a challenge, have a query, or just want a friendly chat, I'm here to assist you! 😊\n"
# # #     "To get started, please enter your Gemini API key below."
# # # )

# # # # Get API key from .env file or user input
# # # api_key = os.getenv("GEMINI_API_KEY")
# # # if not api_key:
# # #     api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

# # # if not api_key:
# # #     st.warning("⚠️ Please add your Gemini API key to continue.", icon="🔐")
# # # else:
# # #     # Create a session state variable to store the chat messages
# # #     if "messages" not in st.session_state:
# # #         st.session_state.messages = []

# # #     # Display the existing chat messages
# # #     for message in st.session_state.messages:
# # #         with st.chat_message(message["role"]):
# # #             st.markdown(message["content"])

# # #     # Create a chat input field
# # #     if prompt := st.chat_input("💬 Ask me anything..."):
# # #         # Store and display the user message
# # #         st.session_state.messages.append({"role": "user", "content": prompt})
# # #         with st.chat_message("user"):
# # #             st.markdown(f"**You:** {prompt}")

# # #         # Prepare API request payload
# # #         url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
# # #         headers = {"Content-Type": "application/json"}
# # #         data = json.dumps({
# # #             "contents": [{"parts": [{"text": prompt}]}]
# # #         })

# # #         # Send request to Gemini API
# # #         response = requests.post(url, headers=headers, data=data)
# # #         if response.status_code == 200:
# # #             response_data = response.json()
# # #             assistant_reply = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
# # #         else:
# # #             assistant_reply = "⚠️ Error: Unable to fetch response from Gemini API. Please check your API key."

# # #         # Display and store assistant response
# # #         with st.chat_message("assistant"):
# # #             st.markdown(f"**Support Bot:** {assistant_reply}")
# # #         st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# # # # Footer with a positive message
# # # st.markdown("---")
# # # st.markdown("🌟 *We're here to support you! Keep striving and keep smiling!* 😊")



# # import streamlit as st
# # import requests
# # import json
# # import os
# # import re
# # from dotenv import load_dotenv

# # # Load environment variables
# # load_dotenv()

# # # Customizing UI
# # st.set_page_config(page_title="EvalAI Chatbot", page_icon="🤖", layout="centered")

# # # Title & Description
# # st.title("🤖 EvalAI Chatbot")
# # st.markdown(
# #     "### Get insights into the EvalAI codebase! 📚\n"
# #     "Ask anything about EvalAI, and this chatbot will fetch relevant details for you. 🚀"
# # )

# # # Get API key
# # api_key = os.getenv("GEMINI_API_KEY")
# # if not api_key:
# #     api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

# # # Initialize session state for document chunks
# # if "doc_chunks" not in st.session_state:
# #     st.session_state.doc_chunks = []
# #     st.session_state.data_loaded = False

# # # Function to chunk text data
# # def create_text_chunks(text_data, chunk_size=2000, overlap=200):
# #     # Clean HTML tags
# #     clean_text = re.sub(r'<[^>]*>', ' ', text_data)
# #     clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
# #     # Create chunks with simple sliding window
# #     chunks = []
# #     for i in range(0, len(clean_text), chunk_size - overlap):
# #         chunk = clean_text[i:i + chunk_size]
# #         chunks.append(chunk)
    
# #     return chunks

# # # Function to find relevant chunks based on keywords
# # def find_relevant_chunks(query, chunks, max_chunks=3):
# #     # Extract keywords (simple approach)
# #     keywords = set(query.lower().split())
# #     # Remove common words
# #     stopwords = {'the', 'is', 'and', 'of', 'in', 'to', 'a', 'for', 'how', 'what', 'why', 'does', 'do'}
# #     keywords = keywords - stopwords
    
# #     # Score chunks based on keyword matches
# #     chunk_scores = []
# #     for i, chunk in enumerate(chunks):
# #         chunk_lower = chunk.lower()
# #         score = sum(keyword in chunk_lower for keyword in keywords)
# #         chunk_scores.append((i, score))
    
# #     # Get top chunks
# #     chunk_scores.sort(key=lambda x: x[1], reverse=True)
# #     top_indices = [idx for idx, score in chunk_scores[:max_chunks] if score > 0]
    
# #     # If no matches found, return first chunk as fallback
# #     if not top_indices:
# #         return [chunks[0]] if chunks else []
    
# #     return [chunks[idx] for idx in top_indices]

# # # Load and chunk data button
# # if not st.session_state.data_loaded and api_key:
# #     if st.button("📥 Load EvalAI Data"):
# #         with st.spinner("Loading and chunking EvalAI data..."):
# #             # Fetch EvalAI Data
# #             evalai_url = "https://uithub.com/Cloud-CV/EvalAI?accept=text%2Fhtml&maxTokens=4096"
# #             evalai_response = requests.get(evalai_url)
            
# #             if evalai_response.status_code == 200:
# #                 evalai_data = evalai_response.text
# #                 # Create chunks
# #                 st.session_state.doc_chunks = create_text_chunks(evalai_data)
# #                 st.session_state.data_loaded = True
# #                 st.success(f"✅ Data loaded and chunked into {len(st.session_state.doc_chunks)} segments!")
# #             else:
# #                 st.error("❌ Failed to fetch EvalAI data.")

# # # Chat interface
# # if api_key:
# #     if "messages" not in st.session_state:
# #         st.session_state.messages = []
    
# #     for message in st.session_state.messages:
# #         with st.chat_message(message["role"]):
# #             st.markdown(message["content"])
    
# #     if prompt := st.chat_input("💬 Ask about EvalAI..."):
# #         st.session_state.messages.append({"role": "user", "content": prompt})
# #         with st.chat_message("user"):
# #             st.markdown(prompt)
        
# #         # Check if data is loaded
# #         if not st.session_state.data_loaded:
# #             with st.chat_message("assistant"):
# #                 response = "Please load the EvalAI data first by clicking the 'Load EvalAI Data' button above."
# #                 st.markdown(response)
# #                 st.session_state.messages.append({"role": "assistant", "content": response})
# #         else:
# #             # Find relevant chunks
# #             relevant_chunks = find_relevant_chunks(prompt, st.session_state.doc_chunks)
# #             relevant_context = "\n\n".join(relevant_chunks)
            
# #             # Trim context if too long
# #             if len(relevant_context) > 12000:  # Limit context size
# #                 relevant_context = relevant_context[:12000] + "..."
            
# #             # Construct prompt with context
# #             enhanced_prompt = f"""
# #             Answer the following question about EvalAI based on this context:
            
# #             Context: {relevant_context}
            
# #             Question: {prompt}
            
# #             If the context doesn't contain enough information to answer the question, 
# #             please respond based on general knowledge about machine learning evaluation platforms.
# #             Keep your answer focused on EvalAI functionality and features.
# #             """
            
# #             # Send query to Gemini API
# #             gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
# #             headers = {"Content-Type": "application/json"}
# #             data = json.dumps({"contents": [{"parts": [{"text": enhanced_prompt}]}]})
            
# #             with st.spinner("Thinking..."):
# #                 gemini_response = requests.post(gemini_url, headers=headers, data=data)
                
# #                 if gemini_response.status_code == 200:
# #                     response_data = gemini_response.json()
# #                     assistant_reply = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
# #                 else:
# #                     assistant_reply = f"⚠️ Error: Unable to fetch response from Gemini API. Status code: {gemini_response.status_code}"
            
# #             with st.chat_message("assistant"):
# #                 st.markdown(assistant_reply)
                
# #             st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
# # else:
# #     st.warning("⚠️ Please add your Gemini API key to continue.", icon="🔐")

# # # Footer
# # st.markdown("---")
# # st.markdown("📘 *Powered by EvalAI Knowledgebase & Gemini AI!* 🚀")




# import streamlit as st
# import json
# import os
# import re
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# st.set_page_config(page_title="EvalAI Chatbot", page_icon="🤖", layout="centered")

# st.title("🤖 EvalAI Chatbot")
# st.markdown(
#     "### Get insights into the EvalAI codebase! 📚\n"
#     "Ask anything about EvalAI, and this chatbot will fetch relevant details for you. 🚀"
# )

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

# data_file_path = "data.txt"
# data_loaded = os.path.exists(data_file_path)

# if "doc_chunks" not in st.session_state:
#     st.session_state.doc_chunks = []
#     st.session_state.data_loaded = False

# def clean_text(text):
#     text = re.sub(r'<[^>]*>', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'http\S+', '', text)
#     return text.strip()

# def search_in_file(file_path, query):
#     relevant_lines = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             if query.lower() in line.lower():
#                 relevant_lines.append(line.strip())
#     return " ".join(relevant_lines)

# def load_and_chunk_text(text_data, chunk_size=1500, overlap=100):
#     clean_data = clean_text(text_data)
#     chunks = [clean_data[i:i + chunk_size] for i in range(0, len(clean_data), chunk_size - overlap)]
#     return chunks

# def refine_search_query(query):
#     gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#     headers = {"Content-Type": "application/json"}
#     prompt = f"Refine this search query for finding relevant information in the EvalAI codebase: {query}"
#     data = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    
#     response = requests.post(gemini_url, headers=headers, data=data)
#     if response.status_code == 200:
#         return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", query)
#     return query

# if not st.session_state.data_loaded:
#     if data_loaded:
#         user_query = st.text_input("Enter search query for relevant data:")
#         if user_query:
#             with st.spinner("Searching and loading relevant data..."):
#                 relevant_text = search_in_file(data_file_path, user_query)
#                 if relevant_text:
#                     st.session_state.doc_chunks = load_and_chunk_text(relevant_text)
#                     st.session_state.data_loaded = True
#                     st.success(f"✅ Loaded {len(st.session_state.doc_chunks)} relevant chunks!")
#                 else:
#                     st.error("❌ No relevant data found in data.txt.")
#     else:
#         st.warning("⚠️ data.txt file not found. Please create this file with your EvalAI data.")

# if api_key:
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     if prompt := st.chat_input("💬 Ask about EvalAI..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         if not st.session_state.data_loaded:
#             response = "Please make sure data.txt exists and contains relevant data."
#         else:
#             context = "\n\n".join(st.session_state.doc_chunks)[:4000]
            
#             enhanced_prompt = f"""
#             Answer the following question about EvalAI based on this context:
            
#             Context: {context}
            
#             Question: {prompt}
            
#             If the context doesn't contain enough information to answer the question, 
#             please respond based on general knowledge about machine learning evaluation platforms.
#             Keep your answer focused on EvalAI functionality and features.
#             """
            
#             gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#             headers = {"Content-Type": "application/json"}
#             data = json.dumps({"contents": [{"parts": [{"text": enhanced_prompt}]}]})
#             gemini_response = requests.post(gemini_url, headers=headers, data=data)
            
#             if gemini_response.status_code == 200:
#                 response = gemini_response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
#             else:
#                 response = f"⚠️ Error: Unable to fetch response from Gemini API. Status code: {gemini_response.status_code}"
        
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
# else:
#     st.warning("⚠️ Please add your Gemini API key to continue.", icon="🔐")

# st.markdown("---")
# st.markdown("📘 *Powered by EvalAI Knowledgebase & Gemini AI!* 🚀")


# import streamlit as st
# import json
# import os
# import re
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# st.set_page_config(page_title="EvalAI Chatbot", page_icon="🤖", layout="centered")

# st.title("🤖 EvalAI Chatbot")
# st.markdown(
#     "### Get insights into the EvalAI codebase! 📚\n"
#     "Ask anything about EvalAI, and this chatbot will fetch relevant details for you. 🚀"
# )

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

# data_file_path = "data.txt"
# data_loaded = os.path.exists(data_file_path)

# if "doc_chunks" not in st.session_state:
#     st.session_state.doc_chunks = []
#     st.session_state.data_loaded = False

# def clean_text(text):
#     text = re.sub(r'<[^>]*>', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'http\S+', '', text)
#     return text.strip()

# def generate_keywords(query):
#     """Uses AI to generate relevant search keywords based on the query."""
#     gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#     headers = {"Content-Type": "application/json"}
#     prompt = f"Generate multiple relevant search keywords for the following query: {query}"
#     data = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
    
#     response = requests.post(gemini_url, headers=headers, data=data)
#     if response.status_code == 200:
#         keywords = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", query)
#         return keywords.split(",")
#     return [query]

# def search_in_file(file_path, queries):
#     relevant_lines = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             if any(query.lower() in line.lower() for query in queries):
#                 relevant_lines.append(line.strip())
#     return " ".join(relevant_lines)

# def load_and_chunk_text(text_data, chunk_size=1500, overlap=100):
#     clean_data = clean_text(text_data)
#     chunks = [clean_data[i:i + chunk_size] for i in range(0, len(clean_data), chunk_size - overlap)]
#     return chunks

# if not st.session_state.data_loaded:
#     if data_loaded:
#         user_query = st.text_input("Enter search query for relevant data:")
#         if user_query:
#             with st.spinner("Searching and loading relevant data..."):
#                 keywords = generate_keywords(user_query)
#                 relevant_text = search_in_file(data_file_path, keywords)
#                 if relevant_text:
#                     st.session_state.doc_chunks = load_and_chunk_text(relevant_text)
#                     st.session_state.data_loaded = True
#                     st.success(f"✅ Loaded {len(st.session_state.doc_chunks)} relevant chunks!")
#                 else:
#                     st.error("❌ No relevant data found in data.txt.")
#     else:
#         st.warning("⚠️ data.txt file not found. Please create this file with your EvalAI data.")

# if api_key:
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     if prompt := st.chat_input("💬 Ask about EvalAI..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         if not st.session_state.data_loaded:
#             response = "Please make sure data.txt exists and contains relevant data."
#         else:
#             context = "\n\n".join(st.session_state.doc_chunks)[:4000]
            
#             enhanced_prompt = f"""
#             Answer the following question about EvalAI based on this context:
            
#             Context: {context}
            
#             Question: {prompt}
            
#             If the context doesn't contain enough information to answer the question, 
#             please respond based on general knowledge about machine learning evaluation platforms.
#             Keep your answer focused on EvalAI functionality and features.
#             """
            
#             gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#             headers = {"Content-Type": "application/json"}
#             data = json.dumps({"contents": [{"parts": [{"text": enhanced_prompt}]}]})
#             gemini_response = requests.post(gemini_url, headers=headers, data=data)
            
#             if gemini_response.status_code == 200:
#                 response = gemini_response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
#             else:
#                 response = f"⚠️ Error: Unable to fetch response from Gemini API. Status code: {gemini_response.status_code}"
        
#         with st.chat_message("assistant"):
#             st.markdown(response)
        
#         st.session_state.messages.append({"role": "assistant", "content": response})
# else:
#     st.warning("⚠️ Please add your Gemini API key to continue.", icon="🔐")

# st.markdown("---")
# st.markdown("📘 *Powered by EvalAI Knowledgebase & Gemini AI!* 🚀")

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