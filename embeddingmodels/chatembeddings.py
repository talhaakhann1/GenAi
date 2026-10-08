
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
     dimensions=64
)

text="how are you"


vector= embedding.embedding_documents(text)

print(vector)