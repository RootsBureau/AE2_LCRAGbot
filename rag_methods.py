#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules["pysqlite3"]

import streamlit as st
import os
import dotenv
import chromadb
import openai
import call_functions as cf

import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.outputs import LLMResult
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyMuPDFLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,  OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 

dotenv.load_dotenv()
DB_DOCS_LIMIT = 10  # Maximum number of documents to load
DB_COLLECTION_LIMIT = 50  # Maximum number of collections in the vector store
# Utility: Animated status dots
DOTS = ["", ".", "..", "..."]

def validate_openai_api_key(api_key: str) -> bool:
    """Check if the provided OpenAI API key is likely valid format."""
    return bool(api_key and api_key.startswith("sk-") and len(api_key) > 20)

#function to streetch theresponse of the LLM
def stream_llm_response(llm_stream, messages):
    response_message = ""
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("🤔 Thinking...")
    
    try:
        for chunk in llm_stream.stream(messages):
            response_message += chunk.content
            yield chunk
    except openai.APIConnectionError as e:
        st.error("Connection to OpenAI failed: " + str(e))

    st.session_state.messages.append({"role": "assistant", "content": response_message})

# -------------------
# -- Indexing phase
# -------------------

def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("./source_files", exist_ok=True)  # Ensure the directory exists                
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())
                   
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyMuPDFLoader(file_path)
                        elif doc_file.name.endswith('.docx'):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Unsupported file type: {doc_file.type}")
                            continue

                        # Load the documents
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                        st.success(f"Document {doc_file.name} loaded successfully.")
                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")

                    finally:
                        os.remove(file_path)

                else:
                    st.error(f"Maximum number of documents ({DB_DOCS_LIMIT}) reached.")
        
        if docs:
            _split_and_add_docs_to_db(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✔️")

# Function to load a URL and add its content to the vector store
def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    raw_docs = loader.load()
                    
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                    st.success(f"URL {url} loaded successfully.")
                except Exception as e:
                    st.toast(f"Error loading URL {url}: {e}", icon=":warning:")
            else:
                st.error(f"Maximum number of URLs/Docs ({DB_DOCS_LIMIT}) reached.")
        if docs:
            _split_and_add_docs_to_db(docs)
            st.toast(f"URL *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✔️")

# Function to initialize the vector store and add documents to it
def initialize_vector_store(docs):
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=st.session_state.openai_api_key),
        collection_name=f"{str(time.time()).replace('.', '')[:14]}_" + st.session_state['session_id'], # Unique collection name based on session ID
        persist_directory="./vector_store"
    )

    # Mamanging the vector store collcetion limit
    chroma_client = chromadb.PersistentClient(path="./vector_store")
    collection_names = [collection.name for collection in chroma_client.list_collections()]
    print("Number of collections:", len(collection_names))
    while len(collection_names) > DB_COLLECTION_LIMIT:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_store


# Function to split documents and add them to the vector store
def _split_and_add_docs_to_db(docs):

    api_key = st.session_state.get("openai_api_key", "")
    if not validate_openai_api_key(api_key):
        st.error("❌ Missing or invalid OpenAI API key. Please set a valid key in the sidebar.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    document_chunk = text_splitter.split_documents(docs)

    # Estimate tokens
    total_tokens = cf.count_tokens_for_embedding([doc.page_content for doc in document_chunk], model=st.session_state["model"])
    st.session_state.total_tokens += total_tokens

    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = initialize_vector_store(docs) # Initialize vector store if not already done
    else:
        st.session_state.vector_store.add_documents(document_chunk) # Add documents to the existing vector store

# -------------------
# -- Retrieving Augmented Generation (RAG) 
# -------------------

def _get_context_retriver_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([                
                ("user", "{input}"),
                ("user", "Given tyhe above conversation, generate a search query to retrieve relevant inforation from documents from the vector store relevent to conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_coversational_rag_chain(llm):
    reteiver_chain = _get_context_retriver_chain(st.session_state.vector_store, llm)

    prompt = ChatPromptTemplate.from_messages([
       ("system", """You are a helpful assistant that answers user questions based on the provided context.
                    For each document, the source is indicated, cite the source in your response using.
                    If content matches use the most relevant information from the documents to answer the user's question. If the content does not match, answer based on your knowledge.\n
                    Context:            
                    {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_document_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(reteiver_chain, stuff_document_chain)    

def stream_llm_rag_response (llm_stream, messages):
    response_message = ""
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("🤔 Thinking...")    
    
    last_input = messages[-1].content.strip()

    # Help function
    if last_input.lower() == "::help":
        response_message = (
            "🛠️ **Available Commands:**\n"
            "- `::list_sources` — List loaded documents\n"
            "- `::summarize_documents` — Summarize all loaded files\n"            
            "- `::summarize_source <filename>` — Summarize defined <filename> document\n"
             "- `::status` — Show documents loased and collection status and limits\n"
            "- `::clear_collections` — Clears vector colection\n"
        )
        st.session_state.messages.append({"role": "assistant", "content": response_message})
        yield response_message
        return

    # RAG status
    if last_input.lower() == "::status":
        info = cf.get_status_info()
        st.session_state.messages.append({"role": "assistant", "content": info})
        yield info
        return
    
    #Clering the vector store collections
    if last_input.lower() == "::clear_collections":
        status_msg = st.empty()
        dots = ["", ".", "..", "..."]
        for i in range(6):
            status_msg.markdown(f"🧹 Clearing collections{dots[i % 4]}")
            time.sleep(0.1)
        result = cf.clear_vector_store_collections()
        status_msg.empty()
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.vector_store = None
        st.session_state.rag_sources = []
        yield result
        return
    
    # List sources
    if last_input.lower() == "::list_sources":
        yield "📚 Loading Sources... Please wait."
        sources = cf.list_sources()
        response_message = "📚 **Loaded Sources:**\n" + "\n".join(f"- {s}" for s in sources)
        st.session_state.messages.append({"role": "assistant", "content": response_message})
        yield response_message
        return

    # Summarize docs all or [filename]
    if last_input.lower().startswith("::summarize_documents") or last_input.lower().startswith("::summarize_source"):
        command_parts = last_input.split(" ", 1)
        filename = command_parts[1].strip() if len(command_parts) > 1 else None

        # Normalize path fragment
        if filename and filename.startswith("./source_files/"):
            filename = os.path.basename(filename)

        # Animated progress placeholder
        status_placeholder = st.empty()
        response_message = None
        for i in range(6):
            status_placeholder.markdown(f"💭 Summarizing, please wait{DOTS[i % 4]}")
            time.sleep(0.1)
            if i == 2:
                try:
                    response_message = cf.summarize_documents(llm_stream, target_source=filename)
                except Exception as e:
                    response_message = f"❗ Error during summarization: {e}"
        status_placeholder.empty()

        st.session_state.messages.append({"role": "assistant", "content": response_message})
        yield response_message
        return

    # --- Default: RAG Q&A ---
    from rag_methods import get_coversational_rag_chain  # optional local import
    conversation_rag_chain = get_coversational_rag_chain(llm_stream)
    response_message = "🔎 RAG'ed Response::\n\n"

    result: dict = conversation_rag_chain.invoke({
        "messages": messages[:-1],
        "input": messages[-1].content
    })
    
    answer = result["answer"]
    sources = set(doc.metadata.get("source", "unknown") for doc in result["context"])
    response_message += result["answer"]
     # Track tokens from LLM output (estimate if metadata not available)
    if hasattr(result, "response_metadata"):
        usage = result.response_metadata.get("token_usage", {})
        model = result.response_metadata.get("model", "Unknown model")
        tokens_used = usage.get("total_tokens", 0)
    else:
        model = "Unknown"
        tokens_used = len(answer.split())  # crude estimate fallback

    response_message += (        
        f"📚 **Sources**:\n" + "\n".join(f"- {src}" for src in sources) + "\n\n"
        f"🧠 **Model**: { st.session_state["model"]} | :1234: **Tokens used**: {tokens_used:,}"
    )

    st.session_state.messages.append({"role": "assistant", "content": response_message})
    st.session_state.total_tokens += tokens_used
    yield response_message