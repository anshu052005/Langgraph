import os
from dotenv import load_dotenv

# PDF load karne ke liye
from langchain_community.document_loaders import PyPDFLoader

# Bade documents ko chhote chunks me todne ke liye
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Groq LLM
from langchain_groq import ChatGroq

# HuggingFace Embeddings (100% Free)
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# --------------------------------------------------------
# .env file load kar rahe hain jisme GROQ_API_KEY stored hai
# --------------------------------------------------------
load_dotenv()

import os

print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# LangSmith Project Name
os.environ["LANGCHAIN_PROJECT"] = "RAG LLM App"

# --------------------------------------------------------
# PDF ka path
# --------------------------------------------------------
PDF_PATH = "langsmith-masterclass-main/islr.pdf"

# ========================================================
# STEP 1 : PDF Load Karna
# ========================================================

# PyPDFLoader PDF ko read karta hai.
# Har page ko ek Document object ke form me load karta hai.

loader = PyPDFLoader(PDF_PATH)

docs = loader.load()

print(f"Total Pages : {len(docs)}")


# ========================================================
# STEP 2 : Text Chunking
# ========================================================

# LLM ek baar me bahut bada text process nahi karta.
# Isliye document ko chhote-chhote pieces (chunks) me divide karte hain.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Har chunk me approx 1000 characters
    chunk_overlap=150     # Pichhle chunk ke 150 chars next chunk me repeat honge
)

splits = splitter.split_documents(docs)

print(f"Total Chunks : {len(splits)}")


# ========================================================
# STEP 3 : HuggingFace Embeddings
# ========================================================

# Embeddings text ko numerical vectors me convert karti hain.
# Similar meaning wale texts ke vectors bhi similar hote hain.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS ek Vector Database hai.
# Ye embeddings ko store karta hai aur similarity search perform karta hai.

vector_store = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

print("Vector Database Created Successfully.")


# ========================================================
# STEP 4 : Retriever
# ========================================================

# Retriever ka kaam hai user ke question ke according
# sabse relevant chunks retrieve karna.

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}   # Top 4 most relevant chunks
)


# ========================================================
# STEP 5 : Prompt Template
# ========================================================

# LLM ko instruction de rahe hain ki
# sirf provided context se answer dena hai.

prompt = ChatPromptTemplate.from_messages([

    (
        "system",

        """
        You are a helpful AI assistant.

        Answer ONLY from the provided context.

        If answer is not present inside the context,
        simply say:

        "I don't know."
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


# ========================================================
# STEP 6 : Groq LLM
# ========================================================

# Ye actual Large Language Model hai
# jo final answer generate karega.

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)


# ========================================================
# STEP 7 : Retrieved Documents ko String me Convert Karna
# ========================================================

# Retriever Documents return karta hai.

# Prompt me bhejne ke liye unhe ek single string banana padta hai.

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)


# ========================================================
# STEP 8 : RunnableParallel
# ========================================================

# RunnableParallel ek hi time par do kaam karega.

# 1. Question ko Retriever ke paas bhejega
# 2. Original Question ko bhi preserve karega

parallel = RunnableParallel({

    "context":

        retriever

        | RunnableLambda(format_docs),

    "question":

        RunnablePassthrough()

})


# ========================================================
# STEP 9 : Final RAG Chain
# ========================================================

# Flow

# User Question

# ↓

# Retriever

# ↓

# Relevant Context

# ↓

# Prompt

# ↓

# Groq LLM

# ↓

# Final Answer

chain = (

    parallel

    | prompt

    | llm

    | StrOutputParser()

)


# ========================================================
# STEP 10 : Ask Question
# ========================================================

print("\nPDF RAG Ready 🚀")

question = input("\nAsk Question : ")

answer = chain.invoke(question)

print("\nAnswer:\n")

print(answer)