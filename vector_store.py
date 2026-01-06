from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from typing import List, Dict
import logging

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """Embeddings using a SentenceTransformer model"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class VectorStoreManager:
    """Manages FAISS vector store for document embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformerEmbeddings(model_name)
        logger.info(f"Initialized SentenceTransformer embedding model: {model_name}")

    def create_vector_store(self, document_chunks: List[Dict[str, str]]) -> FAISS:
        """Create a FAISS vector store from document chunks"""
        try:
            documents = [
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
                for chunk in document_chunks
            ]
            logger.info(f"Creating vector store for {len(documents)} document chunks...")
            vector_store = FAISS.from_documents(documents, self.embedding_model)
            logger.info(f"Successfully created vector store with {len(documents)} documents")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise Exception(f"Failed to create vector store: {str(e)}")

    def add_documents_to_store(self, vector_store: FAISS, new_chunks: List[Dict[str, str]]) -> FAISS:
        """Add new document chunks to existing vector store"""
        try:
            new_documents = [
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
                for chunk in new_chunks
            ]
            vector_store.add_documents(new_documents)
            logger.info(f"Added {len(new_documents)} new documents to vector store")
            return vector_store
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise Exception(f"Failed to add documents to vector store: {str(e)}")

    def search_similar_documents(self, vector_store: FAISS, query: str, k: int = 4) -> List[Dict]:
        """Search for similar documents in the vector store"""
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ]
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise Exception(f"Failed to search documents: {str(e)}")