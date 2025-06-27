import streamlit as st
import os
import dotenv

from langchain_core.prompts import ChatPromptTemplate


def list_sources():
    if "rag_sources" in st.session_state:
        return st.session_state.rag_sources
    return []

def summarize_documents(llm, target_source=None):
    if "vector_store" not in st.session_state:
        return "❗ No documents loaded."

    # Get all chunks and their metadata
    vectorstore = st.session_state.vector_store
    raw_docs = vectorstore._collection.get(include=["documents", "metadatas"])

    # Group chunks by source
    source_chunks = {}
    for doc_text, metadata in zip(raw_docs["documents"], raw_docs["metadatas"]):
        source = metadata.get("source", "unknown")
        filename = os.path.basename(source)
        if target_source and filename != target_source:
            continue
        source_chunks.setdefault(filename, []).append(doc_text)

    if not source_chunks:
        return f"❗ No documents found matching source: `{target_source}`" if target_source else "❗ No documents to summarize."

    # Summarize each grouped document
    summaries = []
    for source, chunks in source_chunks.items():
        full_doc = "\n\n".join(chunks)
        filename = os.path.basename(source)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following document in 2–3 sentences."),
            ("user", full_doc)
        ])
        chain = prompt | llm

        try:
            summary = chain.invoke({})
            if hasattr(summary, "content"):
                summary = summary.content
            elif not isinstance(summary, str):
                summary = str(summary)
        except Exception as e:
            summary = f"⚠️ Failed to summarize: {e}"

        summaries.append(f"📄 Summary of **{filename}**:\n\n{summary.strip()}")

    return "\n\n".join(summaries)