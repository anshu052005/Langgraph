import os
from dotenv import load_dotenv

# LangSmith ke custom tracing ke liye
from langsmith import traceable

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Document ko chunks me todne ke liye
from langchain_text_splitters import RecursiveCharacterTextSplitter
# HuggingFace Embeddings (100% Free)
from langchain_huggingface import HuggingFaceEmbeddings

# Groq LLM
from langchain_groq import ChatGroq

# Vector Database
from langchain_community.vectorstores import FAISS

# LangChain Components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# Environment Variables Load
# =====================================================

# .env file se GROQ API aur LangSmith variables load honge

load_dotenv()

# LangSmith Project Name
os.environ["LANGCHAIN_PROJECT"] = "PDF RAG App"


# =====================================================
# PDF File Path
# =====================================================

PDF_PATH = "langsmith-masterclass-main/islr.pdf"


# =====================================================
# STEP 1 : PDF Loading
# =====================================================

# @traceable ka use isliye kar rahe hain taki
# LangSmith me is function ki execution separately trace ho.

@traceable(name="load_pdf")

def load_pdf(path: str):

    loader = PyPDFLoader(path)

    # Har page ek Document object banega

    return loader.load()


# =====================================================
# STEP 2 : Chunking
# =====================================================

# LLM directly bade documents process nahi karta.

# Isliye document ko multiple overlapping chunks me divide karte hain.

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


# =====================================================
# STEP 3 : Embeddings + FAISS
# =====================================================

# Embeddings text ko numerical vectors me convert karti hain.

# Hum HuggingFace ki free embedding model use kar rahe hain.

@traceable(name="build_vectorstore")

def build_vectorstore(splits):

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    # FAISS vector database create kar rahe hain.

    vectorstore = FAISS.from_documents(

        splits,

        embeddings

    )

    return vectorstore


# =====================================================
# STEP 4 : Complete Setup
# =====================================================

# Ye function complete RAG setup banata hai.

# Isme PDF Load

# ↓

# Chunking

# ↓

# Embedding

# ↓

# FAISS

# sab sequentially execute hote hain.

@traceable(name="setup_pipeline")

def setup_pipeline(pdf_path):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vectorstore = build_vectorstore(splits)

    return vectorstore


# =====================================================
# STEP 5 : Groq LLM
# =====================================================

# Ye actual language model hai
# jo final answer generate karega.

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)


# =====================================================
# STEP 6 : Prompt
# =====================================================

prompt = ChatPromptTemplate.from_messages([

    (

        "system",

        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not available inside the context,

simply reply:

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


# =====================================================
# STEP 7 : Documents Formatting
# =====================================================

# Retriever Documents return karega.

# Prompt me bhejne se pehle unhe ek string banana padega.

def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )


# =====================================================
# STEP 8 : Build Vector Store
# =====================================================

# Yaha pura setup execute hoga.

vectorstore = setup_pipeline(PDF_PATH)


# =====================================================
# STEP 9 : Retriever
# =====================================================

# Retriever ka kaam hai user ke question ke according

# Top-K similar chunks nikalna.

retriever = vectorstore.as_retriever(

    search_type="similarity",

    search_kwargs={

        "k":4

    }

)


# =====================================================
# STEP 10 : RunnableParallel
# =====================================================

# RunnableParallel ek hi time par do kaam karega.

# 1. Retriever se Context lana.

# 2. Original Question ko preserve rakhna.

parallel = RunnableParallel({

    "context":

        retriever

        | RunnableLambda(format_docs),

    "question":

        RunnablePassthrough()

})


# =====================================================
# STEP 11 : Final RAG Chain
# =====================================================

# Complete Flow

# User Question

# ↓

# Retriever

# ↓

# Relevant Chunks

# ↓

# Prompt

# ↓

# Groq LLM

# ↓

# Output Parser

# ↓

# Final Answer

chain = (

    parallel

    | prompt

    | llm

    | StrOutputParser()

)


# =====================================================
# STEP 12 : Ask User Question
# =====================================================

print("PDF RAG Ready 🚀")

question = input("\nAsk Question : ").strip()


# =====================================================
# STEP 13 : LangSmith Run Configuration
# =====================================================

# Ye run LangSmith dashboard me

# "pdf_rag_query"

# naam se show hoga.

config = {

    "run_name": "pdf_rag_query"

}


# =====================================================
# STEP 14 : Execute Chain
# =====================================================

answer = chain.invoke(

    question,

    config=config

)

print("\nAnswer:\n")

print(answer)