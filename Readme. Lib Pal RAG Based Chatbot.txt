🤖 Lib-Pal: RAG-Based Chatbot for Document-Aware Q&A

Lib-Pal is a Retrieval-Augmented Generation (RAG) based chatbot designed for libraries, researchers, and academic users. It allows users to upload local documents and ask natural language questions, receiving accurate, source-grounded answers powered by vector search and Google Gemini API.

This project focuses on local document intelligence, data privacy, and explainable responses with citations—making it suitable for academic libraries and research environments.

🚀 Features

📄 Upload PDF, DOCX, and TXT documents

🔍 Semantic search using vector embeddings (FAISS)

💬 Conversational Q&A over uploaded documents

📚 Source citations with relevance scores

🧠 Powered by Google Gemini LLM

🖥️ Simple and interactive Streamlit UI

🔐 API key via environment variables (secure)

♻️ Reset knowledge base and chat history anytime

🏗️ System Architecture (RAG Workflow)

Document Upload (User)

Text Extraction & Chunking

Embedding Generation (Sentence Transformers)

Vector Storage (FAISS)

Query Embedding

Top-K Similarity Retrieval

Context Injection

Answer Generation (Gemini API)

Response + Source Display

🧩 Project Structure
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── utils/
│   ├── document_processor.py  # PDF, DOCX, TXT parsing & chunking
│   ├── vector_store.py        # FAISS vector store manager
│   ├── rag_pipeline.py        # Retrieval + generation logic
│   └── gemini_client.py       # Google Gemini API interface
├── .env                       # Environment variables (not committed)
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/lib-pal-rag-chatbot.git
cd lib-pal-rag-chatbot

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🔑 Environment Configuration

Create a .env file in the root directory:

GEMINI_API_KEY=your_google_gemini_api_key_here


⚠️ The application will not process documents without a valid API key.

▶️ Run the Application
streamlit run app.py


Then open your browser at:

http://localhost:8501

🧪 Supported File Formats

✅ PDF (.pdf)

✅ Word (.docx)

✅ Plain Text (.txt)

📊 Output & Transparency

Answers are grounded only in uploaded documents

Each response includes:

Retrieved document chunks

Relevance scores

Reduces hallucinations compared to standalone LLMs

🎯 Use Cases

📚 Academic & University Libraries

🔍 Research Assistance & Literature Review

🏛️ Institutional Repositories

🎓 Teaching & Information Literacy

🧠 Local Knowledge Bases (Privacy-Preserving)

🛠️ Technologies Used

Streamlit – UI framework

FAISS – Vector similarity search

Sentence Transformers – Text embeddings

Google Gemini API – Large Language Model

Python-dotenv – Environment management

👨‍💻 Developer

Pawan Pal
Research Scholar (Library & Information Science)
University of Calcutta
Assistant, Central Library, Assam University (Silchar)

Developed with a vision for future-ready, AI-powered academic libraries.

📜 License

This project is intended for educational and research purposes.
You may modify and extend it with proper attribution.
