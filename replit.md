# RAG Chatbot with Gemini API

## Overview

This is a Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit that allows users to upload documents and ask questions about their content. The system processes documents into searchable chunks, stores them in a vector database, and uses Google's Gemini API to generate contextually relevant answers based on retrieved document segments.

## System Architecture

The application follows a modular RAG architecture with clear separation of concerns:

### Frontend Architecture
- **Streamlit Web Interface**: Simple, interactive web app for document upload and chat functionality
- **Session State Management**: Maintains conversation history and component states across user interactions
- **Progress Indicators**: Real-time feedback during document processing

### Backend Architecture
- **Document Processing Pipeline**: Handles multiple file formats (PDF, DOCX, TXT) with chunking strategy
- **Vector Store**: FAISS-based similarity search using sentence transformers for embeddings
- **RAG Pipeline**: Orchestrates document retrieval and response generation
- **LLM Integration**: Google Gemini API for natural language generation

## Key Components

### Document Processor (`utils/document_processor.py`)
- **Purpose**: Extract and chunk text from uploaded documents
- **Supported Formats**: PDF, DOCX, TXT
- **Chunking Strategy**: Fixed-size chunks (1000 chars) with overlap (200 chars) for context preservation
- **Metadata Tracking**: Source file, chunk ID, and file type for each text segment

### Vector Store Manager (`utils/vector_store.py`)
- **Purpose**: Create and manage document embeddings for similarity search
- **Technology**: FAISS (Facebook AI Similarity Search) with sentence-transformers
- **Embedding Model**: all-MiniLM-L6-v2 for efficient text embeddings
- **Search Method**: Cosine similarity with L2 normalization

### RAG Pipeline (`utils/rag_pipeline.py`)
- **Purpose**: Coordinate document retrieval and answer generation
- **Retrieval**: Top-k similar documents based on query embedding
- **Context Preparation**: Concatenates relevant chunks within token limits
- **Generation**: Structured prompts for Gemini API with context injection

### Gemini Client (`utils/gemini_client.py`)
- **Purpose**: Interface with Google Gemini API for text generation
- **Model**: gemini-2.5-flash for balanced performance and cost
- **Configuration**: Configurable temperature and token limits
- **Error Handling**: Graceful fallbacks for API failures

## Data Flow

1. **Document Upload**: User uploads files through Streamlit interface
2. **Text Extraction**: Document processor extracts text based on file type
3. **Chunking**: Text is split into overlapping segments with metadata
4. **Embedding**: Vector store manager creates embeddings using sentence transformers
5. **Storage**: FAISS index stores vectors for fast similarity search
6. **Query Processing**: User questions are embedded and matched against document vectors
7. **Retrieval**: Top relevant chunks are retrieved based on similarity scores
8. **Generation**: RAG pipeline sends context and query to Gemini API
9. **Response**: Generated answer is displayed with source references

## External Dependencies

### Core Technologies
- **Streamlit**: Web framework for rapid prototyping of data applications
- **FAISS**: High-performance vector similarity search library
- **Sentence Transformers**: Pre-trained models for text embeddings
- **Google Gemini API**: Large language model for text generation

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX file processing
- **Built-in Python**: TXT file handling

### Environment Requirements
- **Python 3.11+**: Modern Python features and performance
- **UV Package Manager**: Fast dependency resolution and management
- **PyTorch CPU**: Backend for sentence transformers (CPU-optimized)

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment Target**: Autoscale for dynamic resource allocation
- **Port Configuration**: Streamlit server on port 5000
- **Process Management**: Parallel workflow execution

### Environment Setup
- **API Key Management**: Environment variables for secure credential storage
- **Configuration**: Streamlit config for headless operation and theming
- **Dependencies**: UV lock file ensures reproducible builds

### Scalability Considerations
- **Stateless Design**: Components can be cached and reused across sessions
- **Memory Management**: FAISS indexes are memory-efficient for moderate document collections
- **API Rate Limiting**: Gemini client includes error handling for API constraints

## User Preferences

Preferred communication style: Simple, everyday language.

## Changelog

Changelog:
- June 22, 2025. Initial setup