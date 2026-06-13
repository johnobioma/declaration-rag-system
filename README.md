
@"
# 📜 Declaration of Independence RAG System

AI-powered semantic search for the Declaration of Independence using LangChain, FAISS, and Streamlit.

## ✨ Features
- ChatGPT-style interface with professional dark theme
- Semantic search with FAISS vector database
- Conversation history and source attribution
- Adjustable number of results (k value)
- Quick question buttons for common queries

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenRouter API key

### Installation

1. Clone the repository
2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Create \`.env\` file with your API key:
\`\`\`
OPENROUTER_API_KEY=your_key_here
\`\`\`

### Run the App
\`\`\`bash
streamlit run project.py
\`\`\`

## 📋 Example Questions
- "What is the purpose of the Declaration?"
- "What grievances are listed against the King?"
- "What natural rights are mentioned?"
- "Who signed this document?"

## 🛠️ Tech Stack
- **Streamlit** - Web UI Framework
- **LangChain** - RAG Framework
- **FAISS** - Vector Similarity Search
- **OpenAI Embeddings** - Semantic Search via OpenRouter

## 📝 License
MIT

## 🤝 Contributing
Feel free to open issues or submit pull requests!
"@ | Out-File -FilePath README.md -Encoding UTF8