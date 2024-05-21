import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import OpenAI,HuggingFaceHub
from langchain_community.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import openai
import os
import pickle

# Load environment variable which has OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
hf_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Streamlit file uploader
file = st.file_uploader("Upload a PDF file", type='pdf')

if file is not None:
    reader = PdfReader(file)
    file_name = file.name[:-4]
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text) 
    st.write(f"Total characters in all text chunks: {sum(len(chunk) for chunk in chunks)}")
    
    pickle_file = f"{file_name}.pkl"
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as file:
            vector_store = pickle.load(file)
        st.write(f"The existing pickle file {pickle_file} is loaded from the disk.")
    else:
        # Using HuggingFace Instruct model
        response = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        # Storing in the FAISS vector database and also into a pickle file with GPU support
        vector_store = FAISS.from_texts(chunks, embedding=response)
        with open(pickle_file, "wb") as file:
            pickle.dump(vector_store, file)

    query = st.text_input(f"Query about the uploaded PDF file {file_name}.pdf:")
    if query:
        docs = vector_store.similarity_search(query=query, k=3)
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.6,
                "repetition_penalty": 1.03,
            },
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)
else:
    st.write("Upload a PDF file to extract text.")
