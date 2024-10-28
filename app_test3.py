import streamlit as st
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv
import docx  # for handling DOCX files
import pptx  # for handling PPTX files
from bs4 import BeautifulSoup  # for handling HTML files

# Load environment variables
load_dotenv()

# Initialize session state for chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Dummy function for loading FAISS index (update with actual index loading)
def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Define related topic suggestions if answer not available
def suggest_related_topics(question, docs):
    keywords = ["cybersecurity", "risk", "defense", "state operations"]
    return ", ".join(kw for kw in keywords if kw in question.lower())

# Define the conversational chain for question-answering
def get_conversational_chain():
    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="Given the following context from a document, answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    
    # Initialize the chat model with necessary parameters
    chat_model = ChatGoogleGenerativeAI(
        model="models/chat-001",  # Update with your model ID
        temperature=0.5  # You can adjust this based on your preference
    )
    
    # Load the question-answering chain with the chat model and prompt
    qa_chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt_template)
    
    return qa_chain

# Main chat function to query document and handle responses
def chat_with_document(user_question, chunk_page_mapping, vector_store):
    docs = vector_store.similarity_search(user_question, k=5)  # Fetch more chunks for broader context
    chain = get_conversational_chain()  # Load the conversational chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Handle out-of-context response with related topics
    if response["output_text"].strip().lower() == "answer is not available in the context.":
        related_topics = suggest_related_topics(user_question, docs)
        if related_topics:
            return f"Answer not available. Try asking about related topics: {related_topics}", None
        return response["output_text"], None

    # Retrieve and format page references
    pages = {chunk_page_mapping[i] for i, doc in enumerate(docs)}
    page_references = ", ".join(f"Page {page}" for page in sorted(pages))
    return response["output_text"], page_references

# Main app interface for chat system
st.title("📝 Document Chat System")
st.subheader("Upload your documents and chat with them!")

# File uploader
uploaded_file = st.file_uploader("Upload files (PDF, TXT, CSV, DOCX, PPTX, HTML)", type=["pdf", "txt", "csv", "docx", "pptx", "html", "htm"])

if uploaded_file:
    st.write("📄 Processing Uploaded Document...")
    chunk_page_mapping = {}  # Dummy dictionary for mapping chunks to pages; replace with actual mapping
    vector_store = load_faiss_index()  # Load FAISS index
    
    st.subheader("💬 Chat with Your Document")
    
    # User input field
    user_question = st.text_input("Enter your question here")
    if user_question:
        # Get answer and references
        answer, references = chat_with_document(user_question, chunk_page_mapping, vector_store)
        
        # Store timestamped chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if references:
            st.session_state.chat_history.append((timestamp, f"Q: {user_question}", f"A: {answer}", f"References: {references}"))
        else:
            st.session_state.chat_history.append((timestamp, f"Q: {user_question}", f"A: {answer}"))

        # Display chat history with collapsible answers
        for i, (timestamp, question, response, *ref) in enumerate(st.session_state.chat_history):
            with st.expander(f"{i+1}. {question} ({timestamp})"):
                st.write(response)
                if ref:
                    st.write(ref[0])

        # Add feedback button for each answer
        feedback_buttons = [st.button(f"Helpful ({i+1})") for i in range(len(st.session_state.chat_history))]
