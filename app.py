import streamlit as st
from helpers.llm_helper import call_llm, Conversation

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "convo" not in st.session_state:
    st.session_state.convo = Conversation()

st.set_page_config(
    page_title="AI Chat AWS",
    page_icon=":sparkles:"
)

user_avatar = "👤"
assistant_avatar = "✨"


st.title("AI Chat AWS")
user_input = st.chat_input("Say something")

if user_input:

    st.session_state.convo.add_block("user", user_input)

    for i in st.session_state.convo.get_blocks():
        if i["role"] == "user":
            avatar = user_avatar
        else:
            avatar = assistant_avatar
            
        with st.chat_message(name=i["role"], avatar=avatar):
            st.write(i["content"][0]["text"])

    streaming_response = call_llm(st.session_state.convo)

    with st.chat_message("assistant", avatar="✨"):
        assistant_message = st.write_stream(streaming_response)

    st.session_state.convo.add_block("assistant", assistant_message)
