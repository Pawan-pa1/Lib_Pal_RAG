import os
import logging
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
        
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY environment variable is not set or is using placeholder value")
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise Exception(f"Failed to initialize Gemini client: {str(e)}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a response using Gemini API"""
        try:
            logger.info("Generating response with Gemini...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    candidate_count=1
                )
            )
            
            if response and response.text:
                logger.info("Successfully generated response")
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return "I apologize, but I wasn't able to generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def generate_chat_response(self, conversation_history: list, current_query: str) -> str:
        """Generate a response considering conversation history"""
        try:
            # Format conversation history for context
            conversation_context = ""
            for msg in conversation_history[-5:]:  # Use last 5 messages for context
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
            
            # Create prompt with context
            full_prompt = f"""Previous conversation:
{conversation_context}

Current question: {current_query}

Please provide a helpful response that takes into account the conversation history."""
            
            return self.generate_response(full_prompt)
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return self.generate_response(current_query)  # Fallback to simple response
    
    def test_connection(self) -> bool:
        """Test the connection to Gemini API"""
        try:
            test_response = self.generate_response("Hello, this is a test. Please respond with 'Connection successful.'")
            return "successful" in test_response.lower()
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False