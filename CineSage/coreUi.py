import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv


load_dotenv()


model = ChatGroq(
    model="openai/gpt-oss-20b"
)


class MovieInfo(BaseModel):
    movie_name: str
    release_year: int | None
    director: str
    cast: List[str]
    genre: List[str]
    language: str
    country: str
    runtime: str
    rating: str
    writers: List[str]
    production_company: str
    main_characters: List[str]
    plot: str
    themes: List[str]
    keywords: List[str]
    quick_summary: str


parser = PydanticOutputParser(
    pydantic_object=MovieInfo
)


prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the paragraph.

        {format_instructions}
        """
    ),
    (
        "human",
        "{paragraph}"
    )
])


st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Movie Paragraph",
    height=250,
    placeholder="Enter your movie paragraph here..."
)


if st.button("Extract Information"):

    final_prompt = prompt_template.invoke(
        {
            "paragraph": paragraph,
            "format_instructions": parser.get_format_instructions()
        }
    )

    response = model.invoke(final_prompt)

    # Raw LLM output
    raw_output = response.content

    # Parse raw output into Pydantic object
    movie = parser.parse(raw_output)

    st.header(movie.movie_name)

    st.subheader("Movie Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Release Year:**", movie.release_year)
        st.write("**Director:**", movie.director)
        st.write("**Language:**", movie.language)
        st.write("**Country:**", movie.country)
        st.write("**Runtime:**", movie.runtime)

    with col2:
        st.write("**Genre:**", ", ".join(movie.genre))
        st.write("**Rating:**", movie.rating)
        st.write("**Production Company:**", movie.production_company)

    st.subheader("Cast")
    st.write(", ".join(movie.cast))

    st.subheader("Writers")
    st.write(", ".join(movie.writers))

    st.subheader("Main Characters")
    st.write(", ".join(movie.main_characters))

    st.subheader("Plot")
    st.write(movie.plot)

    st.subheader("Themes")
    st.write(", ".join(movie.themes))

    st.subheader("Keywords")
    st.write(", ".join(movie.keywords))

    st.subheader("Quick Summary")
    st.write(movie.quick_summary)

    # Raw model response
    st.subheader("Raw Model Output")

    st.code(raw_output)