from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import time

app = Flask(__name__)

# === Global Setup ===
PDF_FOLDER = "pdf"
DB_FOLDER = "db"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# LLM & Embeddings
cached_llm = Ollama(model="gemma:2b")  # Use lighter model for speed
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
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information, say so. [/INST]</s>
    [INST] {input} 
           Context: {context}
           Answer: 
    [/INST]
    """
)

# === Persistent Vector Store & Retrieval Chain ===
print("🔄 Loading vector store from disk...")
vector_store = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding)

print("🔄 Creating retrieval chain...")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 20, "score_threshold": 0.3}
)
document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# === Routes ===

@app.route("/pdf", methods=["POST"])
def upload_pdf():
    try:
        file = request.files["file"]
        filename = file.filename
        filepath = os.path.join(PDF_FOLDER, filename)
        file.save(filepath)
        print(f"📄 Uploaded file: {filename}")

        loader = PDFPlumberLoader(filepath)
        docs = loader.load_and_split()
        print(f"📄 Document pages: {len(docs)}")

        chunks = text_splitter.split_documents(docs)
        print(f"📚 Text chunks: {len(chunks)}")

        Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=DB_FOLDER)
        print(f"✅ Document embedded and stored.")

        return jsonify({
            "status": "Successfully uploaded",
            "filename": filename,
            "doc_len": len(docs),
            "chunks_len": len(chunks)
        })

    except Exception as e:
        print(f"❌ Error uploading PDF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai", methods=["POST"])
def ai_query():
    try:
        query = request.json.get("query", "")
        print(f"🤖 Query: {query}")
        response = cached_llm.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        print(f"❌ AI query error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    try:
        query = request.json.get("query", "")
        print(f"📥 PDF query: {query}")
        start = time.time()
        result = retrieval_chain.invoke({"input": query})
        elapsed = time.time() - start
        
        # Safe printing (avoid serialization issues)
        print(f"✅ Done in {elapsed:.2f} seconds")
        print(f"Answer: {result.get('answer')}")
        print(f"Context documents: {len(result.get('context', []))}")
        
        # Return only the serializable answer
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
