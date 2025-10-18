from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- LangChain and AI stack ---
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import (
    PDFPlumberLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, UnstructuredFileLoader, UnstructuredCSVLoader
)

from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate


# --- Utils and app dependencies ---
from cachetools import TTLCache, cached
from sqlalchemy.orm import Session
from database import get_db
from models import User, UserQuery
from pydantic import BaseModel
import os
import time
import hashlib
from dotenv import load_dotenv
import bcrypt


# --- Load environment variables ---
load_dotenv()

# --- FastAPI setup ---
app = FastAPI(title="AI Company Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Password helpers ---
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- Folders setup ---
PDF_FOLDER = "pdf"
DB_FOLDER = "db"
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# --- LLM setup ---
cached_llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

# --- Prompt template ---
raw_prompt = PromptTemplate.from_template("""
<s>[INST]
You are an AI assistant for company-wide information. Answer **using only the provided context**.

Guidelines:
- Answer in plain text only.
- Structure clearly with numbers, paragraphs, or headings.
- If context doesn't contain the answer, say: "This information is not covered in available company documentation".
[/INST]</s>

[INST]
Department: Various
Question: {input}
Available Documentation: {context}

Company Response:
[/INST]
""")

# --- Vector store setup ---
print("🔄 Loading vector store...")
vector_store = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Helpers ---
qa_cache = TTLCache(maxsize=500, ttl=3600)

def file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_file_extension_loaders(filename, filepath):
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
    return loader.load_and_split()

@cached(cache=qa_cache)
def cached_query(query: str):
    print(f"🔍 Querying LLM: {query}")
    return retrieval_chain.invoke({"input": query})

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = None
    contact: str = None
    fullname: str

class LoginRequest(BaseModel):
    username: str
    password: str

# --- Routes ---

@app.post("/AI/ai")
def ai_query(payload: QueryRequest):
    try:
        query = payload.query
        from_cache = query in qa_cache
        result = cached_query(query)
        answer = result.get("answer", "").replace("[INST]", "").replace("[/INST]", "").strip()
        return {"answer": answer, "cached": from_cache}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AI/upload_docs")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = file.filename
        filepath = os.path.join(PDF_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())

        print(f"📄 Uploaded file: {filename}")
        doc_hash = file_hash(filepath)

        if os.path.exists(f"{DB_FOLDER}/{doc_hash}.cached"):
            return {"status": "File already processed", "filename": filename, "doc_hash": doc_hash}

        docs = get_file_extension_loaders(filename, filepath)
        chunks = text_splitter.split_documents(docs)

        global vector_store
        vector_store.add_documents(chunks)  # ✅ No persist() needed

        with open(f"{DB_FOLDER}/{doc_hash}.cached", "w") as f:
            f.write("cached")

        return {"status": "Successfully uploaded", "filename": filename, "chunks": len(chunks)}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AI/ask_docs")
def ask_pdf(payload: QueryRequest):
    try:
        query = payload.query
        start = time.time()
        result = cached_query(query)
        elapsed = time.time() - start
        print(f"✅ Done in {elapsed:.2f}s")
        answer = result.get("answer", "").replace("[INST]", "").replace("[/INST]", "").strip()
        return {"answer": answer, "time": f"{elapsed:.2f}s"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AI/register")
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(User.username == request.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")

        user = User(
            username=request.username,
            email=request.email,
            contact=request.contact,
            password_hash=hash_password(request.password),
            full_name=request.fullname
        )

        db.add(user)
        db.commit()
        db.refresh(user)
        return {"message": "User registered", "username": user.username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AI/login")
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if verify_password(request.password, user.password_hash):
        return {"message": "Login successful", "username": user.username}
    return JSONResponse(status_code=401, content={"message": "Invalid credentials"})
