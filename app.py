import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response
)

dotenv.load_dotenv()

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
]

st.set_page_config(
    page_title="The PMP exam assistant",
    page_icon=":books:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -- Header --
st.html("""<h2 style='text-align: center;'> :books: The PMP Exam Assistant :robot:</h2>""")

# -- Initial Setup --
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "model" not in st.session_state:
    st.session_state.model = MODELS[0]  # Default to the first model

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
            "role" : "assistant",
            "content":"Hi there! Ask questions about the PMBOK Guide or from PMP exam. I'll provide you with the detailed information and sources."
        }
    ]

# -------------
# -- Sidebar --
# -------------
with st.sidebar:
    deafult_openai_api_key = os.getenv("OPENAI_API_KEY", "") if os.getenv("OPENAI_API_KEY") is not None else ""
    with st.popover(":lock: OpenAI"):
        openai_api_key = st.text_input(
            "Add OpenAI API Key:",
            value=deafult_openai_api_key,
            type="password",
            placeholder="sk-...",
            help="Get your OpenAI API key from https://platform.openai.com/account/api-keys",
        )
    
    default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
    with st.popover(":lock: Anthropic"):
        anthropic_api_key = st.text_input(
            "Add Anthropic API Key:",
            value=default_anthropic_api_key,
            type="password",
            placeholder="sk-...",
            help="Get your Anthropic API key from https://console.anthropic.com/account/api-keys",
        )

    st.header("Settings")
    #st.selectbox("Select Model", MODELS)

    st.markdown("---")
    st.markdown("### Session ID")
    st.markdown(f"**{st.session_state.session_id}**")

# -------------
# -- Main Content
# -------------

# check for API keys
missing_openai_key = openai_api_key == "" or openai_api_key is None
missing_anthropic_key = anthropic_api_key == "" or anthropic_api_key is None
if missing_openai_key and missing_anthropic_key:
    st.warning(":attention: Please provide at least one API key (OpenAI or Anthropic) to use the assistant.")

else:
    #-- sidebar
    with st.sidebar:
        st.divider()
        st.selectbox(
            "Select Model",
            [model for model in MODELS if ("openai" in model and not missing_openai_key) or ("anthropic" in model and not missing_anthropic_key)],
            key="model"
        )

        cols0 = st.columns(2)
        # RAG toggle 
        with cols0[0]:
            is_vector_store_loaded = ("vector_store" in st.session_state and st.session_state.vector_store is not None)
            st.toggle(
                "Use RAG",
                value=is_vector_store_loaded,
                key="use_rag",
                help="Enable or disable the use of RAG (Retrieval-Augmented Generation) for answering questions.",
                disabled=not is_vector_store_loaded
            )
        # Clear chat button
        with cols0[1]:
            st.button("clear chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        # File upload for documents
        st.header("RAG Sources")
        st.file_uploader(
            ":file: Upload documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
            help="Drag & drop document (pdf, txt, md, docx) to its content in llm response.",
        )

        # URL input for RAG
        st.text_input(
            ":link: Add URL",
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
            help="Add a URL to load its content into the vector store for RAG.",
        )

        with st.expander(f":document: RAG Sources({0 if not is_vector_store_loaded else len(st.session_state.vector_store.get()['metadatas'])})"):
            st.write([] if not is_vector_store_loaded else [meta["source"] for meta in st.session_state.vector_store.get()["metadatas"]])

        

# -------------
# -- Chat Interface
# -------------

# model selection provider
model_provider = st.session_state.model.split("/")[0]
if model_provider == "openai":
    llm_stream = ChatOpenAI(
        model=st.session_state.model.split("/")[-1],        
        temperature=0.3,
        streaming=True,
    )
elif model_provider == "anthropic":
    llm_stream = ChatAnthropic(
        model_name=st.session_state.model.split("/")[-1],
        temperature=0.3,
        streaming=True,
        timeout=20,  # Set to desired timeout in seconds
        stop=None    # Or provide a list of stop sequences if needed
    )

# previouse message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#initialize the users prompt and add it to the session state last message
if prompt := st.chat_input("Your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))




