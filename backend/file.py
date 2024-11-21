from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def load_file():
    # This is a long document we can split up.
    with open("Data/data.txt") as f:
        data = f.read()
    return [data]

def chunker():
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="gradient"
    )
    docs=text_splitter.create_documents(load_file())
    texts=text_splitter.split_documents(docs)
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(texts,embeddings)
    retriever =vectorstore.as_retriever()
    return retriever



llm=ChatOpenAI(model="gpt-4o")
def create_chain():
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

    Medical Question Criteria:
    - Related to health conditions
    - Involving medical treatments
    - Discussing medical procedures
    - Addressing symptoms or diagnoses
    - Exploring medical recommendations

    Unacceptable Questions:
    - Non-medical general knowledge queries
    - Personal advice outside medical context
    - Speculative or hypothetical medical scenarios
    - Questions without clear medical relevance

    Context Constraints:
    - Extract medical information verbatim
    - Do not interpolate or generate external medical knowledge
    - Maintain the exact medical terminology from the context

    Current Medical Context: {context}
    """


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    human_prompt="How much weight should I loose according to my doctor  ?"
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(chunker(), question_answer_chain)
    results = rag_chain.invoke({"input":  human_prompt })
    print(results)

create_chain()