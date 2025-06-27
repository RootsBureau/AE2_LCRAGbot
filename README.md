# AE2_LCRAGbot

status: work in progress

AE2_LCRAGbot is a RAG suporting AI chat bot designed to add you documents to llm.
This repository contains the source code, configuration, and documentation for setting up and running the bot.

# Demo 

https://rag-ed-assistant-ai.streamlit.app/

## Features

- LLm chantbot with model selection option
- Upload documanent or URL to include informaton in llm responses context (RAG)
- Functions calling support:
     🛠️ Available Commands:"
            ::list_sources` — List loaded documents
            ::summarize_documents` — Summarize all loaded files
            ::summarize_source <filename>` — Summarize defined <filename> document
            ::status` — Show documents loased and collection status and limits
            ::clear_collections` — Clears vector colection

- upcoming:
    1. model usage output
    2. Prompt templates for specific use cases
    3. Progress indicators

## Getting Started

### Prerequisites

- Python 3
- Required dependencies listed in `requirements.txt`

### Installation

Use main Banch to launch loccaly.
Streamlit app Branch is configured for Streamlit could deployment

```bash
git clone https://github.com/yourusername/AE2_LCRAGbot.git
cd AE2_LCRAGbot
pip install -r requirements.txt
```

### Usage

```bash
streamlit run main.py
```

## Configuration

TBD

## Contributing

This is a learnign exersise project
