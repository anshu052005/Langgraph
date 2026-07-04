import os
from dotenv import load_dotenv

# LangSmith tracing decorator
from langsmith import traceable

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace Embeddings (Free)
from langchain_huggingface import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# FAISS Vector Database
from langchain_community.vectorstores import FAISS

# LangChain Components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# ==========================================================
# STEP 1 : Load Environment Variables
# ==========================================================

# .env file se API Keys load hongi
# Isme GROQ_API_KEY aur LangSmith variables hone chahiye.

load_dotenv()

# LangSmith Project Name
os.environ["LANGCHAIN_PROJECT"] = "PDF RAG App2"

# PDF File Path
PDF_PATH = "langsmith-masterclass-main/islr.pdf"


# ==========================================================
# STEP 2 : PDF Load Function
# ==========================================================

# Ye function PDF ko read karega.
# Har page ek Document object ban jayega.

# @traceable lagane se LangSmith me
# is function ki alag tracing dikhegi.

@traceable(name="load_pdf")
def load_pdf(path: str):

    loader = PyPDFLoader(path)

    return loader.load()


# ==========================================================
# STEP 3 : Document Chunking
# ==========================================================

# Bade documents ko directly LLM ko nahi bhejte.

# Isliye unhe multiple overlapping chunks me todte hain.

@traceable(name="split_documents")
def split_documents(
    docs,
    chunk_size=1000,
    chunk_overlap=150
):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap

    )

    return splitter.split_documents(docs)


# ==========================================================
# STEP 4 : Embedding + FAISS
# ==========================================================

# HuggingFace Embeddings completely free hain.

# Ye har chunk ko numerical vector me convert karegi.

# FAISS un vectors ko efficiently store karega.

@traceable(name="build_vectorstore")
def build_vectorstore(splits):

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    vectorstore = FAISS.from_documents(

        documents=splits,

        embedding=embeddings

    )

    return vectorstore


# ==========================================================
# STEP 5 : Complete Setup Pipeline
# ==========================================================

# Ye ek Parent Function hai.

# Iske andar

# PDF Load

# ↓

# Chunking

# ↓

# Embedding

# ↓

# Vector Database

# sequentially execute honge.

# LangSmith me ye sab setup_pipeline ke under grouped dikhenge.

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(

    pdf_path,

    chunk_size=1000,

    chunk_overlap=150

):

    docs = load_pdf(pdf_path)

    splits = split_documents(

        docs,

        chunk_size,

        chunk_overlap

    )

    vectorstore = build_vectorstore(splits)

    return vectorstore


# ==========================================================
# STEP 6 : Groq LLM
# ==========================================================

# Ye actual AI Model hai
# jo final answer generate karega.

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)


# ==========================================================
# STEP 7 : Prompt Template
# ==========================================================

# System Prompt LLM ko instruction deta hai.

# Hum bol rahe hain ki

# sirf Context se answer dena.

prompt = ChatPromptTemplate.from_messages([

    (

        "system",

        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not available inside the context,

reply only:

I don't know.
"""

    ),

    (

        "human",

        """
Question:

{question}

Context:

{context}
"""

    )

])


# ==========================================================
# STEP 8 : Retrieved Documents Formatting
# ==========================================================

# Retriever Documents return karega.

# Prompt me bhejne ke liye unhe ek string banana zaruri hai.

def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )


# ==========================================================
# STEP 9 : Root LangSmith Run
# ==========================================================

# Ye pura RAG Workflow ka Top Level Function hai.

# LangSmith me iske andar

# setup_pipeline

# Retrieval

# Prompt

# LLM

# sab nested dikhenge.

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(

    pdf_path,

    question

):

    # Vector Database Create

    vectorstore = setup_pipeline(

        pdf_path,

        chunk_size=1000,

        chunk_overlap=150

    )


    # Retriever banaya

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={

            "k":4

        }

    )


    # Parallel Execution

    # Ek side Context

    # Dusri side Original Question

    parallel = RunnableParallel({

        "context":

            retriever

            | RunnableLambda(format_docs),

        "question":

            RunnablePassthrough()

    })


    # Complete RAG Chain

    chain = (

        parallel

        | prompt

        | llm

        | StrOutputParser()

    )


    # LangSmith me Query Run Name

    config = {

        "run_name":"pdf_rag_query"

    }


    # Final Answer

    return chain.invoke(

        question,

        config=config

    )


# ==========================================================
# STEP 10 : CLI
# ==========================================================

# User se Question lena

# aur Final Answer print karna.

if __name__ == "__main__":

    print("PDF RAG Ready 🚀")

    question = input("\nAsk Question : ").strip()

    answer = setup_pipeline_and_query(

        PDF_PATH,

        question

    )

    print("\nAnswer:\n")

    print(answer)