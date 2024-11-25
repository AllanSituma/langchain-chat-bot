from dis import Instruction
from itertools import chain
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from htmlTemplates import css, bot_template, user_template
from langchain_huggingface import HuggingFaceEndpoint

outputparser = StrOutputParser()
load_dotenv(find_dotenv())

token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables")

# Print the token to verify it's loaded correctly (remove this line after verifying)
print(f"Token: {token}")

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    chat_history = []

    try:
        response = st.session_state.conversation.invoke({'question': user_question, "chat_history": chat_history})
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")

def get_conversation_chain(vectorstore):
    model = "meta-llama/Meta-Llama-3-8B"

    try:
        llm = HuggingFaceEndpoint(repo_id=model, max_length=128, temperature=0.7, token=token)
    except Exception as e:
        st.error(f"Error initializing HuggingFaceEndpoint: {str(e)}")
        return None

    retriever = vectorstore.as_retriever()

    instruction_to_system = """
    Given a chat history and the latest user question which might
    reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. DO NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """

    qa_system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, provide a summary of the context. DO NOT generate your answer.
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    question_maker_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction_to_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    question_chain = question_maker_prompt | llm | StrOutputParser()

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return question_chain
        else:
            return input["question"]

    retriever_chain = RunnablePassthrough.assign(
        context=contextualized_question | retriever
    )

    rag_chain = (
        retriever_chain
        | qa_prompt
        | llm
    )

    return rag_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple pdfs :books:")
    user_question = st.text_input("Ask a question about your documents")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()