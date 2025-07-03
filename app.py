from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI 
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import ( PDFPlumberLoader , UnstructuredPowerPointLoader , UnstructuredWordDocumentLoader ,
    UnstructuredExcelLoader , UnstructuredFileLoader , UnstructuredCSVLoader)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from cachetools import TTLCache, cached
import os
import time
import requests
import hashlib
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


app = Flask(__name__)
CORS(app , resources={r"/*": { "origins":"*" }}, supports_credentials=True)  # Enable CORS for all routes

# === Global Setup ===
PDF_FOLDER = "pdf"
DB_FOLDER = "db"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# get dynamic retriver
def get_dynamic_retriever(query:str , base_k=3 , max_k=10):
    """
    Returns a retriver with a dynamic k based on query lenght
    longer queries get more context
    """
    query_length = len(query.split())

    # define ranges ( tune as neeeded)
    if query_length <= 5:
        k = base_k
    elif query_length <= 10:
        k = base_k + 1
    elif query_length <= 20:
        k = base_k + 2
    else:
        k = max_k

    print(f"🔍 Dynamic retriever: k={k} for query length {query_length}")

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(k*2, 10),  # Fetch more candidates to ensure quality
        }
    )


# LLM & Embeddings
#cached_llm = Ollama(model="llama3:8b-instruct-q4_0", base_url="http://ollama:11434")
#cached_llm = Ollama(model="llama3:8b", base_url="http://ollama:11434")
#cached_llm = Ollama(model="llama3", base_url="http://ollama:11434")
cached_llm = ChatOpenAI(
    model="llama3-8b-8192",  # or another GROQ-supported model like mixtral-8x7b-32768
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

embedding = FastEmbedEmbeddings()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False
)

# Prompt template
raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST]
    You are a helpful and knowledgeable assistant trained to answer questions in a wide range of academic and professional fields, including:

    - Scientific solutions (physics, chemistry, etc.)
    - Biological and medical explanations
    - Mathematical reasoning and problem-solving
    - Legal insights and summaries
    - Historical facts, timelines, and analysis

    Use the provided context below to answer the user's question. If the context is insufficient, use your general knowledge and reasoning skills to provide a clear, helpful response.

    If you're completely unsure or the information is not available, respond with:
    "The answer is not available in the provided information."

    When relevant, format your answer clearly:
    - Use bullet points for lists
    - Use equations for math and science
    - Provide step-by-step reasoning for problem-solving
    - Keep the language professional, informative, and precise
    [/INST]</s>
    [INST]
    Question: {input}
    Context: {context}
    Answer:
    [/INST]
    """
)



# Load or create vector store on startup
print("🔄 Loading vector store from disk...")
vector_store = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding)

print("🔄 Creating retrieval chain...")
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,
                   "fetch_k": 10}  # Fetch more candidates to ensure quality
)

document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# file hashing function
def file_hash(filepath):
    """Generate a hash for the file to check if it has changed."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
# === Routes ===

# caching LLM responses
qa_cache = TTLCache(maxsize=500, ttl=3600)  # Cache for 5 minutes

#function to determine file extension
def get_file_extension_loaders(filename , filepath):
    """Get the file extension from the filename."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        loader = PDFPlumberLoader(filepath)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(filepath)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(filepath)
    elif ext == ".xlsx":
        loader = UnstructuredExcelLoader(filepath)
    elif ext == ".csv":
        loader = UnstructuredCSVLoader(filepath)
    elif ext in [".txt", ".md"]:
        loader = UnstructuredFileLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    docs = loader.load_and_split()
    
    return docs


@cached(cache=qa_cache)
def cached_query(query):
    """Cached query function to avoid repeated LLM calls."""
    print(f"🔍 Querying LLM: {query}")
    return retrieval_chain.invoke({"input": query})


@app.route("/AI/ai", methods=["POST"])
def ai_query():
    try:
        query = request.json.get("query", "")
        from_cache = query in qa_cache

        print(f"🤖 AI query: {query} (cached={from_cache})")
        result = cached_query(query)  # Same cached function as /AI/ask_pdf
        answer = result.get("answer") if isinstance(result, dict) else result
        answer = answer.replace("[INST]", "").replace("[/INST]", "").strip()

        return jsonify({
            "answer": answer,
            "cached": from_cache
        })

    except Exception as e:
        print(f"❌ AI query error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/AI/upload_docs", methods=["POST"])
def upload_pdf():
    try:
        file = request.files["file"]
        filename = file.filename
        filepath = os.path.join(PDF_FOLDER, filename)
        file.save(filepath)
        print(f"📄 Uploaded file: {filename}")

        doc_hash = file_hash(filepath)
        print(f"🔍 File hash: {doc_hash}")
            
        # Check if the file already exists and has not changed
        if os.path.exists(f"{DB_FOLDER}/{doc_hash}.cached"):
            print("✅ Document already cached. Skipping embedding.")
            return jsonify({
                "status": "File already processed",
                "filename": filename,
                "doc_hash": doc_hash
            })


        #loader = PDFPlumberLoader(filepath)
        #docs = loader.load_and_split()
        try:
            docs = get_file_extension_loaders(filename, filepath)
        except ValueError as e:
           return jsonify({"error": str(e)}), 400
        
        print(f"📄 Document pages: {len(docs)}")
    
        chunks = text_splitter.split_documents(docs)
        print(f"📚 Text chunks: {len(chunks)}")

        # Append new documents to the existing vector store
        global vector_store
        vector_store.add_documents(chunks)
        vector_store.persist()
        print(f"✅ Document embedded and stored (appended).")
        
        with open(f"{DB_FOLDER}/{doc_hash}.cached", "w") as f:
            f.write("cached")

        return jsonify({
            "status": "Successfully uploaded",
            "filename": filename,
            "doc_len": len(docs),
            "chunks_len": len(chunks)
        })

    except Exception as e:
        print(f"❌ Error uploading docs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/AI/ask_docs", methods=["POST"])
def ask_pdf():
    try:
        query = request.json.get("query", "")
        print(f"📥 docs query: {query}")
        start = time.time()
        result = cached_query(query)  # ✅ Caching applied here
        elapsed = time.time() - start

        print(f"✅ Done in {elapsed:.2f} seconds")
        print(f"Answer: {result.get('answer')}")
        print(f"Context documents: {len(result.get('context', []))}")

        answer_text = result.get("answer") if isinstance(result, dict) else result

        return jsonify({"answer": answer_text.replace("[INST]", "").replace("[/INST]", "").strip()})
    
    except Exception as e:
        print(f"❌ docs error: {e}")
        return jsonify({"error": str(e)}), 500


# === Start the app ===
def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
