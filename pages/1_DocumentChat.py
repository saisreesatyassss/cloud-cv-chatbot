# import streamlit as st
# import requests
# import os
# import json
# import re
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Set Streamlit page config
# st.set_page_config(page_title="Ask from Document", page_icon="📄")

# st.title("📄 Ask Questions from a Document Link")

# # Get API key from environment or input
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")


# # Chunk text for Gemini
# def chunk_text(text, chunk_size=1500, overlap=100):
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# # Fetch raw text from a document link
# def fetch_document_text_from_link(link):
#     try:
#         response = requests.get(link)
#         if response.status_code == 200:
#             return response.text
#         else:
#             return None
#     except Exception:
#         return None

# # Start App
# if api_key:
#     doc_link = st.text_input("🔗 Paste document link (raw .txt, .md, .html, etc.)")

#     if doc_link:
#         with st.spinner("📥 Fetching and cleaning document..."):
#             doc_text = fetch_document_text_from_link(doc_link)
            
#             if doc_text:

#                 if doc_text:
#                     chunks = chunk_text(doc_text)
#                     st.success(f"✅ Document fetched and cleaned. {len(chunks)} chunks prepared.")
                    
#                     if "doc_messages" not in st.session_state:
#                         st.session_state.doc_messages = []

#                     for message in st.session_state.doc_messages:
#                         with st.chat_message(message["role"]):
#                             st.markdown(message["content"])

#                     if prompt := st.chat_input("💬 Ask a question about this document..."):
#                         st.session_state.doc_messages.append({"role": "user", "content": prompt})
#                         with st.chat_message("user"):
#                             st.markdown(prompt)

#                         context = "\n\n".join(chunks)[:4000]
#                         enhanced_prompt = f"""
#                         Based on the document below, answer the question.

#                         Document Content:
#                         {context}

#                         Question:
#                         {prompt}

#                         If the document does not contain a clear answer, reply accordingly.
#                         """

#                         gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
#                         headers = {"Content-Type": "application/json"}
#                         data = json.dumps({"contents": [{"parts": [{"text": enhanced_prompt}]}]})

#                         response = requests.post(gemini_url, headers=headers, data=data)
#                         if response.status_code == 200:
#                             answer = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
#                         else:
#                             answer = f"⚠️ Error: Gemini API returned {response.status_code}"

#                         with st.chat_message("assistant"):
#                             st.markdown(answer)

#                         st.session_state.doc_messages.append({"role": "assistant", "content": answer})
#                 else:
#                     st.error("⚠️ Document is empty after cleaning. Make sure the content is valid text.")
#             else:
#                 st.error("❌ Could not fetch the document. Make sure it's a valid raw text-accessible URL.")
# else:
#     st.warning("Please enter your Gemini API Key above to continue.", icon="🔐")


import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import json
import re
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Doc Chat Assistant", page_icon="📚", layout="wide")

st.title("📚 Documentation Chat Assistant")
st.markdown("Ask questions about documentation websites like EvalAI")

# Get API key from environment or input
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("🔑 Enter your Gemini API Key", type="password", placeholder="Your API Key here...")

# Initialize session states
if "doc_content" not in st.session_state:
    st.session_state.doc_content = {}  # {url: content}
if "doc_messages" not in st.session_state:
    st.session_state.doc_messages = []
if "base_url" not in st.session_state:
    st.session_state.base_url = ""
if "indexing_complete" not in st.session_state:
    st.session_state.indexing_complete = False

# Check if URL belongs to the same domain as the base URL
def is_same_domain(url, base_url):
    return urlparse(url).netloc == urlparse(base_url).netloc

# Extract text content from HTML with improved extraction
def extract_text_from_html(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()
    
    # Try multiple selectors to find main content based on common documentation frameworks
    main_content = None
    
    # Check for ReadTheDocs specific content
    main_content = soup.select_one('div.document div.body') or soup.select_one('div.rst-content div[role="main"]')
    
    # Try other common documentation frameworks
    if not main_content:
        main_content = (
            soup.select_one('article.content') or 
            soup.select_one('main.content') or
            soup.select_one('div.documentation') or
            soup.select_one('div.main-content') or
            soup.select_one('div#content') or
            soup.select_one('main') or
            soup.select_one('article') or
            soup.body
        )
    
    if main_content:
        # Get the title
        title_tag = soup.find('title')
        title = title_tag.get_text() if title_tag else url.split('/')[-1] or "Home"
        
        # Try to get any headings to structure content
        headings = main_content.find_all(['h1', 'h2', 'h3'])
        heading_texts = [h.get_text(strip=True) for h in headings if h.get_text(strip=True)]
        
        # Extract text from main content
        paragraphs = main_content.find_all(['p', 'li', 'div.highlight', 'pre', 'code'])
        text_parts = []
        
        # Include title and headings
        if title:
            text_parts.append(f"# {title}")
        
        if heading_texts:
            text_parts.append("## Main Sections:")
            text_parts.extend([f"- {h}" for h in heading_texts[:5]])  # Limit to first 5 headings
        
        # Get paragraph text
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 10:  # Skip very short fragments
                text_parts.append(text)
        
        # Clean up the text
        content = "\n\n".join(text_parts)
        
        # Add metadata
        metadata = f"Page Title: {title}\nURL: {url}\n\n"
        
        return metadata + content
    
    return f"URL: {url}\n\nNo content could be extracted."

# Find all links on a page
def find_links(html_content, base_url, current_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Skip empty links and anchor-only links
        if not href or href.startswith('#'):
            continue
            
        full_url = urljoin(current_url, href)
        
        # Only follow links to the same domain and avoid media files
        if (is_same_domain(full_url, base_url) and
            not full_url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip'))):
            # Normalize URL by removing fragments
            normalized_url = full_url.split('#')[0]
            links.append(normalized_url)
    
    return links

# Chunk text for API limits
def chunk_text(text, chunk_size=5000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Generate answer using Gemini API with improved prompt
def generate_answer(question, contexts, urls):
    try:
        # Add URLs to contexts
        contexts_with_sources = []
        for i, (content, url) in enumerate(zip(contexts, urls)):
            contexts_with_sources.append(f"Source {i+1} ({url}):\n{content}")
        
        # Combine contexts but limit to a reasonable size
        combined_context = "\n\n---\n\n".join(contexts_with_sources)
        chunks = chunk_text(combined_context, 8000)  # Ensure we don't exceed token limits
        context_text = chunks[0]  # Use first chunk for simplicity
        
        prompt = f"""
        You are a helpful documentation assistant for EvalAI, a platform for AI challenges. Answer the user's question based ONLY on the provided documentation content below.
        
        Search query: "{question}"
        
        Documentation Content:
        {context_text}
        
        Instructions:
        1. If the answer is found in the documentation, provide a clear and concise response.
        2. Always cite the specific source (Source 1, Source 2, etc.) when providing information.
        3. If the information isn't in the provided content, say "I couldn't find specific information about this in the documentation" and suggest related terms to search for.
        4. Focus on being accurate rather than comprehensive.
        5. If the documentation mentions code or commands, include them in your answer.
        
        Answer the query: {question}
        """

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})

        response = requests.post(gemini_url, headers=headers, data=data)
        if response.status_code == 200:
            answer = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
            return answer
        else:
            return f"⚠️ Error: Gemini API returned {response.status_code}"
    
    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"

# Main application logic
if api_key:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        doc_url = st.text_input("🔗 Documentation URL", placeholder="https://evalai.readthedocs.io/en/latest/")
        
        if doc_url and doc_url != st.session_state.base_url:
            # Reset if URL changes
            st.session_state.base_url = doc_url
            st.session_state.doc_content = {}
            st.session_state.indexing_complete = False
            
            crawl_progress = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔍 Indexing documentation pages..."):
                # Start with the base URL
                links_to_crawl = [doc_url]
                crawled_urls = set()
                
                max_pages = 30  # Limit to prevent excessive crawling
                
                while links_to_crawl and len(crawled_urls) < max_pages:
                    current_url = links_to_crawl.pop(0)
                    
                    # Normalize URL
                    current_url = current_url.split('#')[0]
                    
                    if current_url in crawled_urls:
                        continue
                    
                    try:
                        response = requests.get(current_url, timeout=10)
                        if response.status_code == 200:
                            crawled_urls.add(current_url)
                            
                            # Extract and store content
                            content = extract_text_from_html(response.text, current_url)
                            st.session_state.doc_content[current_url] = content
                            
                            # Find new links
                            new_links = find_links(response.text, st.session_state.base_url, current_url)
                            for link in new_links:
                                if link not in crawled_urls and link not in links_to_crawl and len(crawled_urls) + len(links_to_crawl) < max_pages:
                                    links_to_crawl.append(link)
                            
                            # Update progress
                            progress = min(len(crawled_urls) / max_pages, 1.0)
                            crawl_progress.progress(progress)
                            status_text.text(f"Indexed {len(crawled_urls)} pages...")
                            
                    except Exception as e:
                        st.error(f"Error crawling {current_url}: {str(e)}")
                        continue
                
                st.session_state.indexing_complete = True
                status_text.text(f"✅ Indexed {len(st.session_state.doc_content)} documentation pages")
    
    with col2:
        if st.session_state.doc_content:
            st.write("📑 Indexed Pages:")
            for i, url in enumerate(list(st.session_state.doc_content.keys())[:10]):  # Show first 10
                page_name = url.split('/')[-1]
                display_name = page_name if page_name else "Home"
                st.write(f"{i+1}. [{display_name}]({url})")
            
            if len(st.session_state.doc_content) > 10:
                st.write(f"...and {len(st.session_state.doc_content) - 10} more pages")
    
    if st.session_state.indexing_complete:
        st.markdown("---")
        st.subheader("💬 Ask about the documentation")
        
        # Display chat history
        for message in st.session_state.doc_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user question
        if prompt := st.chat_input("Ask a question about the documentation..."):
            st.session_state.doc_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching documentation..."):
                    # More sophisticated search
                    search_results = []
                    
                    # Clean up the prompt
                    search_query = prompt.lower()
                    search_terms = set([term for term in re.split(r'\W+', search_query) if len(term) > 2])
                    
                    # Add important domain terms if they appear in the prompt
                    domain_terms = ["challenge", "host", "submit", "evaluation", "team", "participant", 
                                  "phase", "dataset", "leaderboard", "metric", "competition"]
                    
                    # Give higher weight to domain terms
                    weighted_terms = search_terms.copy()
                    for term in domain_terms:
                        if term in search_query:
                            weighted_terms.add(term)
                            weighted_terms.add(term)  # Add twice for higher weight
                    
                    # Search through content
                    for url, content in st.session_state.doc_content.items():
                        content_lower = content.lower()
                        
                        # Calculate relevance score
                        # 1. Exact query match (highest weight)
                        exact_matches = content_lower.count(search_query) * 10
                        
                        # 2. Term matches
                        term_matches = sum(content_lower.count(term) for term in weighted_terms)
                        
                        # 3. Title match bonus
                        title_match = 0
                        page_name = url.split('/')[-1]
                        if page_name:
                            page_name_lower = page_name.lower()
                            for term in weighted_terms:
                                if term in page_name_lower:
                                    title_match += 5
                        
                        total_score = exact_matches + term_matches + title_match
                        if total_score > 0:
                            search_results.append((url, content, total_score))
                    
                    # Sort by relevance
                    search_results.sort(key=lambda x: x[2], reverse=True)
                    
                    # Debug: show search results
                    if not search_results:
                        st.session_state.debug_msg = f"No matches found for: {search_terms}"
                    
                    # Get top contexts
                    contexts = []
                    context_urls = []
                    for url, content, _ in search_results[:3]:  # Top 3 most relevant
                        contexts.append(content)
                        context_urls.append(url)
                    
                    # If no results, use introduction/index pages
                    if not contexts:
                        for url, content in st.session_state.doc_content.items():
                            if "index" in url or "intro" in url or url.endswith("/"):
                                contexts.append(content)
                                context_urls.append(url)
                                if len(contexts) >= 2:
                                    break
                    
                    answer = generate_answer(prompt, contexts, context_urls)
                    st.markdown(answer)
            
            st.session_state.doc_messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Please enter your Gemini API Key above to continue.", icon="🔐")

# Add footer
st.markdown("---")
st.caption("Powered by Gemini AI • Built with Streamlit")