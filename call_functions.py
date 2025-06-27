import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate

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
