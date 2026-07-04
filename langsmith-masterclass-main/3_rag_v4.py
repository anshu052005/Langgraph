import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "PDF RAG App3"

PDF_PATH = "islr.pdf"

INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# --------------------------------------------------------
# PDF Loading
# --------------------------------------------------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()


# --------------------------------------------------------
# Chunking
# --------------------------------------------------------

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


# --------------------------------------------------------
# Build Vector Store
# --------------------------------------------------------

@traceable(name="build_vectorstore")
def build_vectorstore(
    splits,
    embed_model_name: str
):

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name
    )

    return FAISS.from_documents(
        splits,
        embeddings
    )


# --------------------------------------------------------
# Fingerprint
# --------------------------------------------------------

def _file_fingerprint(path: str):

    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:

        for chunk in iter(lambda: f.read(1024 * 1024), b""):

            h.update(chunk)

    return {

        "sha256": h.hexdigest(),

        "size": p.stat().st_size,

        "mtime": int(p.stat().st_mtime)

    }


# --------------------------------------------------------
# Cache Key
# --------------------------------------------------------

def _index_key(
    pdf_path,
    chunk_size,
    chunk_overlap,
    embed_model_name
):

    meta = {

        "pdf_fingerprint": _file_fingerprint(pdf_path),

        "chunk_size": chunk_size,

        "chunk_overlap": chunk_overlap,

        "embedding_model": embed_model_name,

        "format": "v1"

    }

    return hashlib.sha256(

        json.dumps(
            meta,
            sort_keys=True
        ).encode("utf-8")

    ).hexdigest()


# --------------------------------------------------------
# Load Saved Index
# --------------------------------------------------------

@traceable(name="load_index", tags=["index"])
def load_index_run(
    index_dir,
    embed_model_name
):

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name
    )

    return FAISS.load_local(

        str(index_dir),

        embeddings,

        allow_dangerous_deserialization=True

    )


# --------------------------------------------------------
# Build New Index
# --------------------------------------------------------

@traceable(name="build_index", tags=["index"])
def build_index_run(

    pdf_path,

    index_dir,

    chunk_size,

    chunk_overlap,

    embed_model_name

):

    docs = load_pdf(pdf_path)

    splits = split_documents(

        docs,

        chunk_size,

        chunk_overlap

    )

    vectorstore = build_vectorstore(

        splits,

        embed_model_name

    )

    index_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    vectorstore.save_local(

        str(index_dir)

    )

    (index_dir / "meta.json").write_text(

        json.dumps(

            {

                "pdf_path": os.path.abspath(pdf_path),

                "chunk_size": chunk_size,

                "chunk_overlap": chunk_overlap,

                "embedding_model": embed_model_name,

            },

            indent=2,

        )

    )

    return vectorstore


# --------------------------------------------------------
# Load or Build
# --------------------------------------------------------

def load_or_build_index(

    pdf_path,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",

    force_rebuild=False

):

    key = _index_key(

        pdf_path,

        chunk_size,

        chunk_overlap,

        embed_model_name

    )

    index_dir = INDEX_ROOT / key

    cache_hit = index_dir.exists() and not force_rebuild

    if cache_hit:

        print("Loading Existing Index...")

        return load_index_run(

            index_dir,

            embed_model_name

        )

    else:

        print("Building New Index...")

        return build_index_run(

            pdf_path,

            index_dir,

            chunk_size,

            chunk_overlap,

            embed_model_name

        )


# --------------------------------------------------------
# Groq LLM
# --------------------------------------------------------

llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0

)


# --------------------------------------------------------
# Prompt
# --------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([

    (

        "system",

        "Answer ONLY from the provided context. If not found, say you don't know."

    ),

    (

        "human",

        "Question: {question}\n\nContext:\n{context}"

    )

])


# --------------------------------------------------------
# Formatter
# --------------------------------------------------------

def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )


# --------------------------------------------------------
# Setup Pipeline
# --------------------------------------------------------

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(

    pdf_path,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",

    force_rebuild=False

):

    return load_or_build_index(

        pdf_path=pdf_path,

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        embed_model_name=embed_model_name,

        force_rebuild=force_rebuild

    )


# --------------------------------------------------------
# Complete RAG Pipeline
# --------------------------------------------------------

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(

    pdf_path,

    question,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",

    force_rebuild=False

):

    vectorstore = setup_pipeline(

        pdf_path,

        chunk_size,

        chunk_overlap,

        embed_model_name,

        force_rebuild

    )

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={

            "k":4

        }

    )

    parallel = RunnableParallel({

        "context":

            retriever

            | RunnableLambda(format_docs),

        "question":

            RunnablePassthrough(),

    })

    chain = (

        parallel

        | prompt

        | llm

        | StrOutputParser()

    )

    return chain.invoke(

        question,

        config={

            "run_name":"pdf_rag_query",

            "tags":["qa"],

            "metadata":{

                "k":4

            }

        }

    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    print("PDF RAG Ready 🚀")

    question = input("\nAsk Question : ").strip()

    answer = setup_pipeline_and_query(

        PDF_PATH,

        question

    )

    print("\nAnswer:\n")

    print(answer)