from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(
    model="openai/gpt-oss-20b"
)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert movie information extractor.

        Your task is to analyze a movie paragraph provided by the user
        and extract useful information from it.

        Extract the following information when available:

        Movie Name:
        Release Year:
        Director:
        Cast:
        Genre:
        Language:
        Country:
        Runtime:
        Rating:
        Writers:
        Production Company:
        Main Characters:
        Plot:
        Themes:
        Keywords:

        After extracting the information, provide:

        Quick Summary:
        Write a short 2–4 sentence paragraph explaining what the movie
        is about, its main conflict, and what makes the story interesting.

        Rules:
        - Only use information provided in the paragraph.
        - Do not make up or assume missing information.
        - If information is not mentioned, write "Not mentioned".
        - Keep the extracted information concise.
        - Keep the Quick Summary natural and easy to understand.
        - Focus on useful information and avoid unnecessary details.
        """
    ),
    (
        "human",
        """
        Extract the movie information from the following paragraph:

        {paragraph}
        """
    )
])

para=input("Give your paragraph :")

final_prompt=prompt_template.invoke(
    {"paragraph":para}
)

response=model.invoke("Hi")

print(response.content)