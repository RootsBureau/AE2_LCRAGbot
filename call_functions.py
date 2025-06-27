import streamlit as st
import os
import dotenv

from langchain_core.prompts import ChatPromptTemplate


def list_sources():
    if "rag_sources" in st.session_state:
        return st.session_state.rag_sources
    return []

def summarize_documents(llm):
    if "vector_store" not in st.session_state:
        return "❗ No documents loaded."

    # Get all chunks with their source metadata
    vectorstore = st.session_state.vector_store
    raw_docs = vectorstore._collection.get(include=["documents", "metadatas"])

    # Step 1: Group chunks by their source
    source_chunks = {}
    for doc_text, metadata in zip(raw_docs["documents"], raw_docs["metadatas"]):
        source = metadata.get("source", "unknown")
        if source not in source_chunks:
            source_chunks[source] = []
        source_chunks[source].append(doc_text)

    # Step 2: Generate a summary for each grouped document
    summaries = []
    for source, chunks in source_chunks.items():
        full_doc = "\n\n".join(chunks)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following document in 2–3 sentences."),
            ("user", full_doc)
        ])
        chain = prompt | llm

        try:
            summary = chain.invoke({})
            if not isinstance(summary, str):
                summary = str(summary)
        except Exception as e:
            summary = f"⚠️ Failed to summarize: {e}"

        summaries.append(f"📄 **{source}**:\n{summary.strip()}")

    return "\n\n".join(summaries)