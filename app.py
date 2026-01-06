# Fixed imports
import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_pipeline import RAGPipeline
from utils.gemini_client import GeminiClient
import google.generativeai as genai
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import ast

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Lib-Pal Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the main components"""
    doc_processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    return doc_processor, vector_store_manager

def get_gemini_client():
    """Get Gemini client, initializing only when API key is available"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        return GeminiClient()
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

doc_processor, vector_store_manager = initialize_components()

def process_uploaded_files(uploaded_files):
    """Process uploaded files and update vector store"""
    # Check if Gemini client can be initialized
    gemini_client = get_gemini_client()
    if not gemini_client:
        st.error("Cannot process documents: Gemini API key is not configured properly.")
        return
    
    with st.spinner("Processing documents..."):
        all_chunks = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Process each file
            chunks = doc_processor.process_file(uploaded_file)
            all_chunks.extend(chunks)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            # Create or update vector store
            vector_store = vector_store_manager.create_vector_store(all_chunks)
            st.session_state.vector_store = vector_store
            
            # Initialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                vector_store=vector_store,
                gemini_client=gemini_client
            )
            
            st.session_state.documents_processed = True
            st.success(f"Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} text chunks!")
        else:
            st.error("No text could be extracted from the uploaded files.")

def main():
    st.title("🤖 Lib-Pal : RAG-Based Chatbot")
    st.markdown("Upload documents and ask questions to get answers with source citations!")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to create your knowledge base"
        )
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)

        # Display current status
        st.header("📊 Status")
        if st.session_state.documents_processed:
            st.success("✅ Documents processed")
            if st.session_state.vector_store:
                doc_count = len(st.session_state.vector_store.docstore._dict)
                st.info(f"📚 {doc_count} chunks in knowledge base")
        else:
            st.warning("⏳ No documents processed yet")

        # API Key status
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            st.success("🔑 API Key configured")
        else:
            st.error("❌ Gemini API Key not configured")
            st.markdown("Please set your `GEMINI_API_KEY` in the environment variables.")

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat Interface")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"*Score: {source['score']:.3f}*")
                                st.markdown(source["content"])
                                st.divider()
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.documents_processed:
                st.error("Please upload and process documents first!")
                return
            if not st.session_state.rag_pipeline:
                st.error("RAG pipeline not initialized. Please process documents first!")
                return
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        start = time.time()
                        response = st.session_state.rag_pipeline.query(prompt)
                        end = time.time()
                        st.markdown(response["answer"])
                        assistant_message = {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["sources"]
                        }
                        st.session_state.messages.append(assistant_message)
                        if response["sources"]:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(response["sources"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                                    st.markdown(source["content"])
                                    if i < len(response["sources"]):
                                        st.divider()
                        latency = end - start
                        st.write(f"Latency: {latency:.3f} seconds")
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    with col2:
        # About section at the top
        st.header("ℹ️ About")
        st.markdown("""
        This Local-RAG based chatbot allows you to:
        - Upload PDF, DOCX, and TXT files
        - Ask questions about your documents
        - Get answers with source citations
        """)

        # Controls section below About
        st.header("🔧 Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        if st.button("🔄 Reset All"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.rag_pipeline = None
            st.session_state.documents_processed = False
            st.rerun()
            
            
        
    # Static colorful footer (fixed at bottom)
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            z-index: 100;
            background: linear-gradient(90deg, #4F8BF9 0%, #1CB5E0 100%);
            color: #fff;
            font-size: 1.5em;
            font-weight: bold;
            padding: 0.5em 0 0.5em 0;
            border-radius: 0;
            box-shadow: 0 -2px 12px rgba(79,139,249,0.18);
            letter-spacing: 1px;
            text-align: center;
        }
        .footer span {
            color: #FFD700;
            font-weight: bold;
            font-size: 1.2em;
        }
        </style>
        <div class="footer">
            Devloped and Maintained by : <span>Pawan Pal</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # (Example metric calculation removed to avoid static/hard-coded values)

if __name__ == "__main__":
    main()
