#import nltk
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Download NLTK data (optional)
#nltk.download('punkt')

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Define embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Load the word document
def load_word_document(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()
    return docs

# Create retrieval function
def create_vector_store(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(documents)
    db_faiss = FAISS.from_documents(chunks, embeddings)
    return db_faiss

# Load and process the document
docs = load_word_document("Declaration of independence.docx")
print(f"Loaded {len(docs)} document(s)")



# Create vector store
db_faiss = create_vector_store(docs, embeddings)
print("Vector store created successfully!")

# Test retrieval
query = "What is the purpose of declaration?"
retrieved_docs = db_faiss.similarity_search(query, k=3)

print(f"\n--- Results for: '{query}' ---\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"Chunk {i}:")
    print(doc.page_content[:400])
    print("-" * 50)