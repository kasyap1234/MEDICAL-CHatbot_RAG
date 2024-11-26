from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import qdrant_client
from qdrant_client.http import models as rest
from qdrant_client.http.models import CollectionStatus, Distance, VectorParams
from typing import Optional, Dict, List
import uuid
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./medical_chatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User database model
class DBUser(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Security configurations
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    context: list

class FileUploadResponse(BaseModel):
    collection_id: str
    message: str

# User management functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(DBUser).filter(DBUser.username == username).first()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
        user = get_user(db, username)
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except jwt.JWTError:
        raise HTTPException(status_code=401)

class MedicalChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        self.qdrant_client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )

    def _create_collection_name(self, user_id: str) -> str:
        return f"medical_data_{user_id}"

    async def create_user_collection(self, user_id: str, file_content: str) -> str:
        collection_name = self._create_collection_name(user_id)
        
        # Create collection with user metadata
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            metadata={
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "type": "medical_data"
            }
        )
        
        # Process and embed documents
        text_splitter = SemanticChunker(self.embeddings)
        docs = text_splitter.create_documents([file_content])
        
        vectors = []
        for idx, doc in enumerate(docs):
            embedding = self.embeddings.embed_query(doc.page_content)
            vectors.append(rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "user_id": user_id,
                    "chunk_id": idx,
                    "metadata": {"source": "user_upload"}
                }
            ))
        
        # Upload vectors to Qdrant
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=vectors
        )
        
        return collection_name

    def get_user_retriever(self, user_id: str):
        collection_name = self._create_collection_name(user_id)
        
        search_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="user_id",
                    match=rest.MatchValue(value=user_id)
                )
            ]
        )
        
        vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings,
            search_filter=search_filter
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    def create_chain(self, retriever):
        system_prompt = """
        You are a specialized medical information assistant designed to:
        - Answer ONLY medical-related questions
        - Provide information STRICTLY from the given medical context
        - Maintain professional and precise medical communication

        IMPORTANT VALIDATION RULES:
        1. Reject non-medical questions immediately
        2. Use ONLY information present in the provided context
        3. If no relevant medical information exists, respond with:
           "I cannot find information about this in the given context."
        4. Be concise, professional, and medically accurate

        Current Medical Context: {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    async def get_medical_response(self, query: str, user_id: str):
        try:
            retriever = self.get_user_retriever(user_id)
            chain = self.create_chain(retriever)
            results = chain.invoke({"input": query})
            return {
                "response": results['answer'],
                "context": [doc.page_content for doc in results['context']]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI Application
app = FastAPI(
    title="Medical RAG Chatbot",
    description="Multi-tenant Medical Information Retrieval and Generation API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Medical Chatbot
medical_chatbot = MedicalChatbot()

@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = pwd_context.hash(user.password)
    new_user = DBUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: DBUser = Depends(get_current_user)
):
    try:
        content = await file.read()
        file_content = content.decode()
        
        collection_name = await medical_chatbot.create_user_collection(current_user.username, file_content)
        
        return FileUploadResponse(
            collection_id=collection_name,
            message="Medical data uploaded and processed successfully"
        )
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: DBUser = Depends(get_current_user)
):
    try:
        response = await medical_chatbot.get_medical_response(request.query, current_user.username)
        return ChatResponse(
            response=response['response'],
            context=response['context']
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
