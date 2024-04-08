import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger

from chains import (
    load_embedding_model,
    load_llm,
)

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "SentenceTransformer" )
llm_name = os.getenv("LLM", "llama2")
url = os.getenv("NEO4J_URI")

# Check if the required environment variables are set
if not all([url, username, password, ollama_base_url]):
    st.write("The application requires some information before running.")
    with st.form("connection_form"):
        url = st.text_input("Enter NEO4J_URI",)
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
            llm_name = "gpt-35-turbo"
            os.environ['AZURE_OPENAI_API_KEY'] = openai_apikey
            os.environ["AZURE_OPENAI_ENDPOINT"] = "https://api-key.openai.azure.com/"

os.environ["NEO4J_URL"] = url
logger = get_logger(__name__)
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

def main():
    st.header("Chat with your file")

    # upload your file
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "csv", "xlsx", "txt"])

    if uploaded_file is not None:
        # Check the file type
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'pdf':
            # Read PDF file
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        elif file_extension in ['csv', 'xlsx']:
            if file_extension == 'csv':
                # Read CSV file
                file_data = pd.read_csv(uploaded_file)
            else:
                # Read Excel file
                file_data = pd.read_excel(uploaded_file)

            # Combine text from all columns
            text = ""
            for column in file_data.columns:
                if file_data[column].dtype == 'object' and not file_data[column].dropna().empty:
                    text += " ".join(file_data[column].dropna().astype(str)) + " "

        elif file_extension == 'txt':
            # Read plain text file
            text = uploaded_file.read().decode("utf-8")

        else:
            st.error("Unsupported file type. Please upload a PDF, CSV, Excel, or text file.")
            return

        # langchain_textsplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Store the chunks part in db (vector)
        vectorstore = Neo4jVector.from_texts(
            chunks,
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="file_bot",
            node_label="FileBotChunk",
            pre_delete_collection=False,  # Delete existing file data
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # Accept user questions/query
        query = st.text_input("Ask questions about your file")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()
