from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage,HumanMessage,BaseMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

messages: list[BaseMessage] = [
    SystemMessage(content="Act as a funny ai asistant")
]


while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot:", response.content)

print(messages)