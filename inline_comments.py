import os

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# Get environment variables
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "SentenceTransformer")
llm_name = os.getenv("LLM", "llama2")

# Prompt user for environment variables if not set
if not all([url, username, password, ollama_base_url]):
    st.write("The application requires some information before running.")
    with st.form("connection_form"):
        url = st.text_input("Enter NEO4J_URI")
        username = st.text_input("Enter NEO4J_USERNAME")
        password = st.text_input("Enter NEO4J_PASSWORD", type="password")
        ollama_base_url = st.text_input("Enter OLLAMA_BASE_URL")
        st.markdown("Only enter the OPENAI_APIKEY to use OpenAI instead of Ollama. Leave blank to use Ollama.")
        openai_apikey = st.text_input("Enter OPENAI_API_KEY", type="password")
        submit_button = st.form_submit_button("Submit")
    if submit_button:
        if not all([url, username, password]):
            st.write("Enter the Neo4j information.")
        if not (ollama_base_url or openai_apikey):
            st.write("Enter the Ollama URL or OpenAI API Key.")
        if openai_apikey:
            llm_name = "gpt-3.5"
            os.environ['OPENAI_API_KEY'] = openai_apikey

os.environ["NEO4J_URL"] = url

# Set up logging
logger = get_logger(__name__)

# Load embedding model
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# Custom callback handler for streaming updates to the UI
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Load language model (LLM)
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# Main function
def main():
    st.header("📄Chat with your pdf file")

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Store text chunks in Neo4j database as vectors
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
            pre_delete_collection=False,  # Delete existing PDF data
        )

        # Initialize question answering (QA) system
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/queries
        query = st.text_input("Ask questions about your PDF file")

        if query:
            # Initialize StreamHandler to update UI with responses
            stream_handler = StreamHandler(st.empty())
            # Run QA system with user query
            qa.run(query, callbacks=[stream_handler])

# Entry point of the script
if __name__ == "__main__":
     main()
