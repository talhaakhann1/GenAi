import streamlit as st

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Store the same message history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages: list[BaseMessage] = [
        SystemMessage(content="Act as a sad ai asistant and reply every message in a sad way")
    ]

st.title("Sad AI Assistant")

# Show chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # User message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)

    # AI response
    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.write(response.content)

     
