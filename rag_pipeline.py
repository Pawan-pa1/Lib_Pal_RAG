from typing import Dict, List
from utils.gemini_client import GeminiClient
from langchain_community.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline that combines document retrieval with Gemini API for generation"""
    
    def __init__(self, vector_store: FAISS, gemini_client: GeminiClient, max_context_length: int = 4000):
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.max_context_length = max_context_length
        logger.info("RAG Pipeline initialized")
    
    def query(self, question: str, num_sources: int = 4) -> Dict:
        """Process a query through the RAG pipeline"""
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving documents for query: {question[:100]}...")
            relevant_docs = self._retrieve_documents(question, num_sources)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": []
                }
            
            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Step 3: Generate answer using Gemini
            logger.info("Generating answer with Gemini...")
            answer = self._generate_answer(question, context)
            
            # Step 4: Format sources
            sources = self._format_sources(relevant_docs)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": []
            }
    
    def _retrieve_documents(self, query: str, k: int) -> List[Dict]:
        """Retrieve relevant documents from vector store"""
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            # Sort by score (higher is better for cosine similarity)
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Retrieved {len(formatted_results)} documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _prepare_context(self, relevant_docs: List[Dict]) -> str:
        """Prepare context string from relevant documents"""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(relevant_docs):
            # Format document with source information
            doc_text = f"[Source {i+1} - {doc['metadata'].get('source', 'Unknown')}]\n{doc['content']}\n"
            
            # Check if adding this document would exceed max length
            if current_length + len(doc_text) > self.max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n".join(context_parts)
        logger.info(f"Prepared context with {len(context_parts)} documents ({len(context)} characters)")
        
        return context
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini API"""
        try:
            # Create a comprehensive prompt
            prompt = f"""Based on the following context from uploaded documents, please answer the user's question. 
If the context doesn't contain enough information to answer the question, please say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate, and helpful answer based on the context
- If information is incomplete or unclear, mention this
- Reference specific sources when possible
- If the context doesn't contain relevant information, state this clearly
- Be conversational but informative

Answer:"""

            answer = self.gemini_client.generate_response(prompt)
            
            if not answer or answer.strip() == "":
                return "I was unable to generate an answer to your question. Please try rephrasing your question or check if your documents contain relevant information."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _format_sources(self, relevant_docs: List[Dict]) -> List[Dict]:
        """Format sources for display"""
        sources = []
        
        for doc in relevant_docs:
            source_info = {
                "content": doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                "score": doc["score"],
                "metadata": doc["metadata"]
            }
            sources.append(source_info)
        
        return sources
    
    def update_vector_store(self, new_vector_store: FAISS):
        """Update the vector store used by the pipeline"""
        self.vector_store = new_vector_store
        logger.info("Vector store updated in RAG pipeline")
