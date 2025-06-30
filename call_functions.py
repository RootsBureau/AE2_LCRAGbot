import streamlit as st
import os
import chromadb
import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from rag_methods import DB_DOCS_LIMIT, DB_COLLECTION_LIMIT


def list_sources():
    return st.session_state.get("rag_sources", [])

def summarize_documents(llm, target_source=None):
    if "vector_store" not in st.session_state:
        return "❗ No documents loaded."

    vectorstore = st.session_state.vector_store
    raw_docs = vectorstore._collection.get(include=["documents", "metadatas"])

    # Group by source
    source_chunks = {}
    for doc_text, metadata in zip(raw_docs["documents"], raw_docs["metadatas"]):
        source = metadata.get("source", "unknown")
        filename = os.path.basename(source)
        if target_source and filename != target_source:
            continue
        source_chunks.setdefault(filename, []).append(doc_text)

    if not source_chunks:
        return f"❗ No documents found matching: `{target_source}`" if target_source else "❗ No documents to summarize."

    summaries = []
    for filename, chunks in source_chunks.items():
        full_doc = "\n\n".join(chunks)
        if len(full_doc) > 12000:
            full_doc = full_doc[:12000] + "\n\n[...truncated]"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following document in 2–3 sentences."),
            ("user", "{input}")
        ])
        chain = prompt | llm

        try:
            summary = chain.invoke({"input": full_doc})
            summary = summary.content if hasattr(summary, "content") else str(summary)
        except Exception as e:
            summary = f"⚠️ Failed to summarize: {e}"

        summaries.append(f"📄 **{filename}**\n\n{summary.strip()}")

    return "\n\n---\n\n".join(summaries)

def clear_vector_store_collections():
    try:
        chroma_client = chromadb.PersistentClient(path="./vector_store")
        collections = chroma_client.list_collections()
        for c in collections:
            chroma_client.delete_collection(name=c.name)
            if "vector_store" in st.session_state:
                st.session_state.vector_store = None
            if "rag_sources" in st.session_state:
                st.session_state.rag_sources = []
        return f"🧹 Cleared {len(collections)} collections."            
    except Exception as e:
        return f"⚠️ Failed to clear collections: {e}"
    



def get_status_info():

    doc_count = len(st.session_state.get("rag_sources", []))

    try:
        chroma_client = chromadb.PersistentClient(path="./vector_store")
        collection_count = len(chroma_client.list_collections())
    except Exception as e:
        collection_count = "❓"

    return (
        f"  📄 **Documents loaded**: {doc_count} / {DB_DOCS_LIMIT}\n\n"
        f"  🧠 **Vector collections**: {collection_count} / {DB_COLLECTION_LIMIT}"
    )

def count_tokens_for_embedding(texts: list[str], model: str = "text-embedding-3-small") -> int:
    """
    Estimate token usage for a list of texts, falling back to a generic encoding if model is unknown.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Safest fallback for most OpenAI models

    total = sum(len(encoding.encode(t)) for t in texts)
    return total