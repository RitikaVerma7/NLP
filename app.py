import os
import openai
import pandas as pd
import faiss
import streamlit as st
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Secure your API key using environment variables or Streamlit secrets in production
openai.api_key = ""
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="RAG-Powered News Chatbot", page_icon="🤖")

# Load dataset and create documents
df = pd.read_csv("TopNewsToday.csv")
df['combined'] = df['Title'] + " " + df['Description']

documents = [
    Document(
        page_content=f"Title: {row['Title']}\nDescription: {row['Description']}\nURL: {row['URL']}\n",
        metadata={"Title": row['Title'], "Description": row['Description'], "URL": row['URL']}
    )
    for _, row in df.iterrows()
]

# Initialize embeddings and vector store
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
vectorstore = FAISS.from_documents(documents, embedding_function)

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key, temperature=0)

# Prompt template
prompt_template = """
You are a friendly and helpful news assistant.

Goals:
1. Ask about user's mood and what kind of news they would like to see.
2. Retrieve relevant news articles from context (derived from CSV). Each article format:
   Title: ...
   Description: ...
   URL: ...
3. If no articles found, inform the user and suggest alternative topics.

Conversation so far:
{chat_history}

User's message: "{question}"

Retrieved context (news articles):
{context}

Instructions:
- Engage the user based on mood or conversation.
- If articles found, present them (Title, Description, URL).
- If no articles found, inform user and suggest another category.
- Always maintain a friendly and helpful tone.
"""

PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_template
)

# Create Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=st.session_state["memory"],
    combine_docs_chain_kwargs={"prompt": PROMPT},
    verbose=False
)

# UI
st.title("🤖 RAG-Powered News Chatbot")
st.write("Tell me how you're feeling or what news you'd like. I’ll share articles from my database. If nothing matches, I'll let you know.")

# Conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", "Hi! How are you feeling today? What kind of news would you like?")]

# Display chat history
for role, message in st.session_state.messages:
    st.chat_message(role).write(message)

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").write(user_input)

    with st.spinner("Fetching news..."):
        start_time = time.time()
        response = chain({"question": user_input})
        answer = response["answer"]
        end_time = time.time()

    st.session_state.messages.append(("assistant", answer))
    st.chat_message("assistant").write(answer)

    # Display response time
    st.sidebar.markdown(f"**Response Time:** {end_time - start_time:.2f} seconds")

# Sidebar: Conversation history
st.sidebar.title("Chat History")
for role, message in st.session_state.messages:
    st.sidebar.markdown(f"**{role.capitalize()}:** {message}")
