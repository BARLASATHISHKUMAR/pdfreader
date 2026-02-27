import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv()
key = os.getenv("gemini_api_key")

# Load PDF
loader = PyPDFLoader(r"C:/Users/barla/Downloads/G1155566631824192_08-02-2026_13-39-59.961_APPLICATION.pdf")
doc = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = splitter.split_documents(doc)

# Create Embeddings (LOCAL - No API errors)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Vector Store
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings
)

vector_store.add_documents(documents)
retriever = vector_store.as_retriever()

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question only from the given context.

    Question:
    {question}

    Context:
    {context}
    """
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    api_key=key,
    model="gemini-2.5-flash",
    temperature=0
)

# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.image("https://innomatics.in/wp-content/uploads/2023/01/innomatics-footer-logo.png")

query = st.text_input("Enter your question")

if st.button("Predict"):
    if query:
        response = rag_chain.invoke(query)
        st.write(response)
    else:
        st.write("Please enter a question.")