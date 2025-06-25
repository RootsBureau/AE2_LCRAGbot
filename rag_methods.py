import streamlit as st

#function to streetch theresponse of the LLM
def stream_llm_response(llm_stream, messages):
    response_message = "thinking..."
    
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})