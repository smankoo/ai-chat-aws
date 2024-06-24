import streamlit as st
from helpers.llm_helper import Conversation
from helpers.invoke_model_converse_stream_api import stream_conversation

from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="AI Chat AWS", page_icon=":sparkles:")
user_avatar = "👤"
assistant_avatar = "✨"

st.title("AI Chat AWS")


if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "convo" not in st.session_state:
    st.session_state.convo = Conversation()
    
if "aws_access_key_id" not in st.session_state:
    if "AWS_ACCESS_KEY_ID" in os.environ:
        st.session_state.aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    
if "aws_secret_access_key" not in st.session_state:
    if "AWS_SECRET_ACCESS_KEY" in os.environ:
        st.session_state.aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []



with st.sidebar:
    st.write("## AWS Credentials")
    st.write("Required permissions: `bedrock:InvokeModelWithResponseStream`")
    st.write("See [docs](https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html)")
    
    if "aws_access_key_id" not in st.session_state:
        aws_access_key_id = st.text_input("AWS Access Key ID", key="aws_access_key_id")

    if "aws_secret_access_key" not in st.session_state:
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key", key="aws_secret_access_key", type="password"
        )

    # hide sidebar by defailt
    hide_menu_style = """
        <style>
            #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def clear_conversation():
    st.session_state.convo = Conversation()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Say something..."):
    # with the user icon, write the question to the front end
    with st.chat_message("user"):
        st.markdown(question)
    # append the question and the role (user) as a message to the session state
    st.session_state.messages.append({"role": "user",
                                      "content": question})
    # respond as the assistant with the answer
    with st.chat_message("assistant"):
        # making sure there are no messages present when generating the answer
        message_placeholder = st.empty()
        # calling the invoke_llm_with_streaming to generate the answer as a generator object, and using
        # st.write stream to perform the actual streaming of the answer to the front end
        answer = st.write_stream(stream_conversation(question))
    # appending the final answer to the session state
    st.session_state.messages.append({"role": "assistant",
                                      "content": answer})