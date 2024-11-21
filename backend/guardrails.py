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

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(chunker(), question_answer_chain)

    results = rag_chain.invoke({"input": "What is the problem doctor is referring to ? "})

    results