from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
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
from typing import Optional, Dict
import uuid
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

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

class User(BaseModel):
    username: str
    email: str
    full_name: str

class UserInDB(User):
    hashed_password: str

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

# Mock user database - Replace with actual database
users_db = {}
user_collections: Dict[str, str] = {}

class MedicalChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        self.qdrant_client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )

    def _create_collection_name(self, user_id: str) -> str:
        """Create a unique collection name for the user"""
        return f"medical_data_{user_id}"

    async def create_user_collection(self, user_id: str, file_content: str) -> str:
        """Create or update user's collection"""
        collection_name = self._create_collection_name(user_id)
        
        # Delete existing collection if it exists
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass

        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="gradient"
        )
        docs = text_splitter.create_documents([file_content])
        
        # Create new collection with user's data
        vectorstore = Qdrant.from_documents(
            docs,
            self.embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
            force_recreate=True
        )
        
        return collection_name

    def get_user_retriever(self, user_id: str):
        """Get retriever for user's collection"""
        collection_name = self._create_collection_name(user_id)
        vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )
        return vectorstore.as_retriever()

    def create_chain(self, retriever):
        """Create RAG chain"""
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
        """Generate medical response using user's data"""
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

# Authentication functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
        return username
    except jwt.JWTError:
        raise HTTPException(status_code=401)

# FastAPI Application
app = FastAPI(
    title="Medical RAG Chatbot",
    description="A medical information retrieval and generation API"
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

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401)
    
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Upload medical document for user"""
    try:
        content = await file.read()
        file_content = content.decode()
        
        collection_name = await medical_chatbot.create_user_collection(current_user, file_content)
        user_collections[current_user] = collection_name
        
        return FileUploadResponse(
            collection_id=collection_name,
            message="Medical data uploaded and processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_class=PlainTextResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: str = Depends(get_current_user)
):
    """Chat with user's medical data"""
    try:
        response = await medical_chatbot.get_medical_response(request.query, current_user)
        return response['response']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
