import openai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
import pickle
# load_dotenv()

# #loading the openai API Key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# embedding_model=OpenAIEmbeddings(model='text-embedding-3-small')

# text_embed=embedding_model.embed_query("This is a sample to check the embedding")
from langchain_community.vectorstores import FAISS
import numpy as np
import pickle

# Assuming your FAISS object has been loaded from a pickle file
pickle_file = "C:\\Users\\abhij\\Masters\\Projects\\DocuMentor\\test.pkl"

with open(pickle_file, "rb") as file:
    vector_store = pickle.load(file)

# Access the FAISS index (assuming it is stored directly as an attribute or similar)
# This part might need adjustment based on how the vector_store object is structured
if hasattr(vector_store, 'index'):
    faiss_index = vector_store.index
else:
    faiss_index = vector_store  # If the vector_store itself is the index

# Retrieving all vectors from the FAISS index
# We assume the index contains N vectors of dimension D
n_vectors = faiss_index.ntotal
vector_dim = faiss_index.d
all_vectors = faiss_index.reconstruct_n(0, n_vectors)

# Print all vectors
print(all_vectors)
print(all_vectors.shape)



