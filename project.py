"""
Professional ChatGPT-style UI for Declaration of Independence RAG System
Modern design with custom colors and professional layout
"""

import streamlit as st
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Custom CSS for ChatGPT-like appearance
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f1724 100%);
        color: #e0e0e0;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #00a8ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Search result styling */
    .result-container {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #2c3e50;
        transition: all 0.3s ease;
    }
    
    .result-container:hover {
        border-color: #00a8ff;
        box-shadow: 0 4px 15px rgba(0,168,255,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1724 0%, #1a1a2e 100%);
        border-right: 1px solid #2c3e50;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00a8ff 0%, #0066cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,168,255,0.4);
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background: #1e2a3a;
        border: 2px solid #2c3e50;
        border-radius: 10px;
        color: white;
        font-size: 16px;
        padding: 10px 15px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00a8ff;
        box-shadow: 0 0 0 2px rgba(0,168,255,0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #00a8ff;
    }
    
    /* Success message styling */
    .stAlert {
        background: #0f1724;
        border: 1px solid #00a8ff;
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1e2a3a;
        border-radius: 8px;
        color: #00a8ff;
        font-weight: 600;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2c3e50;
    }
    
    [data-testid="stMetric"] label {
        color: #00a8ff !important;
    }
    
    [data-testid="stMetric"] value {
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2a3a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00a8ff;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0066cc;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #00a8ff 0%, #0066cc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        margin-top: 0;
        margin-bottom: 30px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: #a0a0a0;
        font-size: 12px;
        border-top: 1px solid #2c3e50;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Declaration RAG System - AI-Powered Search",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom title with gradient effect
st.markdown('<p class="main-title">📜 Declaration of Independence</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Semantic Search • Ask Anything About America\'s Founding Document</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Cache vector store loading
@st.cache_resource
def load_vector_store():
    """Load or create the vector store"""
    with st.spinner("🔮 Initializing AI engine..."):
        # Load document
        loader = UnstructuredWordDocumentLoader("Declaration of independence.docx")
        docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create vector store
        db_faiss = FAISS.from_documents(chunks, embeddings)
        
        return db_faiss, len(chunks)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    st.markdown("---")
    
    k = st.slider("📊 Number of results", min_value=1, max_value=10, value=3, 
                  help="How many relevant passages to retrieve")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    st.markdown("Click any to ask instantly:")
    
    quick_questions = [
        "🎯 What is the purpose of the Declaration?",
        "👑 What grievances are listed against the King?",
        "✨ What natural rights are mentioned?",
        "✍️ Who signed this document?",
        "⚡ Why did the colonies separate from Britain?",
        "📖 What does it say about equality?",
        "🏛️ How is the document structured?"
    ]
    
    for q in quick_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.current_query = q
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.markdown("---")
    
    # Try to load and show status
    try:
        db_faiss, num_chunks = load_vector_store()
        st.success(f"✅ System Ready")
        st.metric("📚 Document Chunks", num_chunks)
        st.metric("🔍 Index Size", db_faiss.index.ntotal)
        st.caption(f"⏱️ Last updated: {datetime.now().strftime('%H:%M:%S')}")
    except FileNotFoundError:
        st.error("❌ Document not found")
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:50]}")

# Main content area - ChatGPT-like conversation
try:
    # Load vector store
    db_faiss, num_chunks = load_vector_store()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Query input (ChatGPT-like)
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Message AI Assistant...",
            placeholder="Ask anything about the Declaration of Independence...",
            label_visibility="collapsed",
            key="user_input",
            value=st.session_state.current_query
        )
    
    with col2:
        search_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    # Clear session query after use
    if st.session_state.current_query:
        st.session_state.current_query = ""
    
    # Process query
    if (search_button or query) and query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message immediately
        st.markdown(f'<div class="user-message">👤 {query}</div>', unsafe_allow_html=True)
        
        # Perform search
        with st.spinner("🔍 Analyzing and retrieving relevant passages..."):
            retrieved_docs = db_faiss.similarity_search(query, k=k)
            
            # Format results as assistant response
            assistant_response = f"**Found {len(retrieved_docs)} relevant passages**\n\n"
            
            for i, doc in enumerate(retrieved_docs, 1):
                # Clean and truncate content
                content = doc.page_content.strip()
                if len(content) > 300:
                    content = content[:300] + "..."
                
                assistant_response += f"**📄 Passage {i}:**\n{content}\n\n"
                assistant_response += f"---\n\n"
            
            # Add metadata summary
            if retrieved_docs:
                sources = list(set([doc.metadata.get('row_index', 'Unknown') for doc in retrieved_docs]))
                assistant_response += f"📊 *Sources: {len(retrieved_docs)} passages retrieved from document*"
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # Display assistant response
            st.markdown(f'<div class="assistant-message">🤖 {assistant_response}</div>', unsafe_allow_html=True)
            
            # Show detailed expandable view
            with st.expander("🔍 View Detailed Results", expanded=False):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Result {i}**")
                    st.markdown(doc.page_content)
                    st.caption(f"Metadata: {doc.metadata}")
                    st.markdown("---")
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a question first.")

    # Footer
    st.markdown('<div class="footer">Powered by LangChain • FAISS Vector Search • OpenAI Embeddings • 🇺🇸 Historical Document RAG System</div>', unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Document 'Declaration of independence.docx' not found!")
    st.info("📁 Please ensure the document is in the current directory: " + os.getcwd())
    st.write("**Current files:**")
    for file in os.listdir("."):
        if file.endswith((".docx", ".doc")):
            st.write(f"📄 {file}")

except Exception as e:
    st.error(f"❌ System Error: {str(e)}")
    st.info("Please check your API key and internet connection.")

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()