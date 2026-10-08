import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
print(os.getenv("HF_TOKEN"))

llm = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=1024,
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("I was just kidding")

print(response.content)
