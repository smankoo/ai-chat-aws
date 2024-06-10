import streamlit as st
from helpers.llm_helper import call_llm, Conversation

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "convo" not in st.session_state:
    st.session_state.convo = Conversation()

st.set_page_config(page_title="AI Chat AWS", page_icon=":sparkles:")

user_avatar = "👤"
assistant_avatar = "✨"

st.title("AI Chat AWS")

with st.sidebar:
    st.write("## AWS Credentials")
    st.write("Required permissions: `bedrock:InvokeModelWithResponseStream`")
    st.write("See [docs](https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html)")
    aws_access_key_id = st.text_input("AWS Access Key ID", key="aws_access_key_id")

    aws_secret_access_key = st.text_input(
        "AWS Secret Access Key", key="aws_secret_access_key", type="password"
    )


def clear_conversation():
    st.session_state.convo = Conversation()


user_input = st.chat_input("Say something...")

if user_input:

    if not aws_access_key_id or not aws_secret_access_key:
        st.info(
            "Please add your AWS Access Key ID and Secret Access Key to \
            continue."
        )
        st.stop()

    st.session_state.convo.add_block("user", user_input)

    for i in st.session_state.convo.get_blocks():
        if i["role"] == "user":
            avatar = user_avatar
        else:
            avatar = assistant_avatar

        with st.chat_message(name=i["role"], avatar=avatar):
            st.write(i["content"][0]["text"])

    try:
        streaming_response = call_llm(
            st.session_state.convo,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    except Exception as e:
        st.error(f"ERROR: Can't invoke LLM model. Reason: {str(e)}")
        st.stop()

    with st.chat_message("assistant", avatar="✨"):
        assistant_message = st.write_stream(streaming_response)

    st.session_state.convo.add_block("assistant", assistant_message)

if st.session_state.convo.get_blocks():
    st.button("Clear", on_click=clear_conversation)
