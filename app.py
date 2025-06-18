from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from cachetools import TTLCache, cached
import os
import time
import requests
import hashlib

app = Flask(__name__)
CORS(app , resources={r"/*": { "origins":"*" }}, supports_credentials=True)  # Enable CORS for all routes

# === Global Setup ===
PDF_FOLDER = "pdf"
DB_FOLDER = "db"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Checking if Ollama is connecting
def wait_for_ollama_ready(url="http://ollama:11434", timeout=5):
    for i in range(timeout):
        try:
            res = requests.get(url + "/")
            if res.status_code == 200:
                print("✅ Ollama is ready!")
                return True
        except:
            pass
        print(f"⏳ Waiting for Ollama... ({i+1}/{timeout})")
        time.sleep(1)
    raise RuntimeError("Ollama not available after timeout.")

wait_for_ollama_ready()

# LLM & Embeddings
#cached_llm = Ollama(model="llama3:8b-instruct-q4_0", base_url="http://ollama:11434")
cached_llm = Ollama(model="llama3:8b", base_url="http://ollama:11434")
#cached_llm = Ollama(model="llama3", base_url="http://ollama:11434")

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
    You are a helpful and knowledgeable technical assistant.
    Use the provided context to answer questions from any subject, including:
    - Mathematics (provide step-by-step solutions),
    - Physics, Chemistry, Biology (explain clearly with reasoning),
    - History and Social Sciences (give factual, well-structured answers),
    - Accounting and Economics (apply relevant formulas or explain terms).

    If the answer is not in the context, say: "The answer is not available in the provided information."

    Be concise, accurate, and use bullet points, equations, or steps where appropriate.
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
    search_type="similarity",
    search_kwargs={"k": 3,}
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

        return jsonify({
            "answer": answer,
            "cached": from_cache
        })

    except Exception as e:
        print(f"❌ AI query error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/AI/pdf", methods=["POST"])
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


        loader = PDFPlumberLoader(filepath)
        docs = loader.load_and_split()
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
        print(f"❌ Error uploading PDF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/AI/ask_pdf", methods=["POST"])
def ask_pdf():
    try:
        query = request.json.get("query", "")
        print(f"📥 PDF query: {query}")
        start = time.time()
        result = cached_query(query)  # ✅ Caching applied here
        elapsed = time.time() - start

        print(f"✅ Done in {elapsed:.2f} seconds")
        print(f"Answer: {result.get('answer')}")
        print(f"Context documents: {len(result.get('context', []))}")

        answer_text = result.get("answer", "No answer found.")
        return jsonify({"answer": answer_text})
    
    except Exception as e:
        print(f"❌ ask_pdf error: {e}")
        return jsonify({"error": str(e)}), 500


# === Start the app ===
def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
