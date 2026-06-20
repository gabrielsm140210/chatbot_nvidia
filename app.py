import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

st.set_page_config(page_title="NVIDIA RAG PDF Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Assistente de Manual em PDF (NVIDIA RAG)")
st.write("Faça perguntas em português sobre o produto com base no manual fornecido.")

nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

if not nvidia_api_key:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        
        pass

if not nvidia_api_key:
    st.info("Por favor, adicione sua NVIDIA_API_KEY no arquivo .env (local) ou nos Secrets do Streamlit (deploy).", icon="🔑")
    st.stop()

@st.cache_resource(show_spinner="Processando o manual em PDF...")
def inicializar_rag():
    nome_arquivo_pdf = "manual.pdf"
   
    if not os.path.exists(nome_arquivo_pdf):
        st.error(f"Arquivo '{nome_arquivo_pdf}' não foi encontrado!")
        st.stop()
   
    loader = PyPDFLoader(nome_arquivo_pdf)
    paginas = loader.load()
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,        
        chunk_overlap=50       
    )
    docs = text_splitter.split_documents(paginas)
   
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )
   
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
   
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = inicializar_rag()

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    nvidia_api_key=nvidia_api_key,
    temperature=0.2
)

template_prompt = """
Você é um assistente técnico especializado e prestativo.
Os fragmentos de contexto abaixo foram extraídos do manual do produto e ESTÃO EM INGLÊS.
Sua tarefa é analisar o contexto em inglês, mas responder à pergunta do usuário OBRIGATORIAMENTE EM PORTUGUÊS.

Use estritamente as informações fornecidas para responder. Se a resposta não puder ser encontrada no texto, diga explicitamente: "Desculpe, mas essa informação não consta no manual do produto."

Contexto (em inglês):
{context}

Pergunta (em português): {question}
Resposta em português:
"""
prompt = ChatPromptTemplate.from_template(template_prompt)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Processei o manual em inglês com sucesso. O que você deseja saber sobre o produto?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt_usuario := st.chat_input("Ex: Qual é o significado do erro 4?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)
       
    with st.chat_message("assistant"):
        with st.spinner("Consultando manual técnico..."):
            try:
                resposta = rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao processar a requisição na API da NVIDIA: {e}")