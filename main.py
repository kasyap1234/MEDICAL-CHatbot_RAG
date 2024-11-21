from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler 
from dotenv import load_dotenv
import logging
from fastapi.responses import PlainTextResponse
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ChatRequest(BaseModel):
    query: str
from fastapi.middleware.cors import CORSMiddleware

# Add this after creating the app



class ChatResponse(BaseModel):
    response: str
    context: list


class MedicalChatbot:
    def __init__(self, document_path: str = "Data/data.txt"):
        """
        Initialize the Medical Chatbot with document retrieval and RAG chain
        """
        self.document_path = document_path
        self.llm = ChatOpenAI(model="gpt-4o")
        self.retriever = self._create_retriever()
        self.chain = self._create_chain()

    def _load_file(self):
        """Load medical context file"""
        try:
            with open(self.document_path, 'r') as f:
                data = f.read()
            return [data]
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise

    def _create_retriever(self):
        """Create a semantic retriever from medical documents"""
        try:
            text_splitter = SemanticChunker(
                OpenAIEmbeddings(),
                breakpoint_threshold_type="gradient"
            )
            docs = text_splitter.create_documents(self._load_file())
            texts = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(texts, embeddings)
            return vectorstore.as_retriever()
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

    def _create_chain(self):
        """Create a retrieval-augmented generation (RAG) chain"""
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
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        return rag_chain

    def get_medical_response(self, query: str):
        """Generate medical response using RAG"""
        try:
            results = self.chain.invoke({"input": query},config={"callbacks": [langfuse_handler]})
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
    description="A medical information retrieval and generation API"
)

# Initialize Medical Chatbot
medical_chatbot = MedicalChatbot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_class=PlainTextResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for medical queries

    :param request: Chat request containing the query
    :return: Medical response text
    """
    try:
        response = medical_chatbot.get_medical_response(request.query)
        return response['response']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint

    :return: Service status
    """
    return {"status": "healthy"}


# Optional: If you want to run this directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
