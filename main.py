import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
import conf

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("ResearchBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

google_api_key = conf.google_api_key

llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=google_api_key, temperature=0.1)

main_placeholder = st.empty()

urls = []

for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

query = st.sidebar.text_input("Question: ")

process_url_clicked = st.sidebar.button("Process URLs")


def process_urls():
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    
    loader  = UnstructuredURLLoader(urls= urls)
    data = loader.load()
    
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    
    spliiter  = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = spliiter.split_documents(data)
    
    vectorstore = Chroma.from_documents(documents=docs, embedding=GooglePalmEmbeddings(google_api_key=google_api_key))
 
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    return vectorstore




def get_answers(vectorstore):
    chain = RetrievalQA.from_chain_type(llm=llm, 
                chain_type="map_reduce", 
                retriever=vectorstore.as_retriever(), 
                input_key="question", return_source_documents=True)
    
    try:
        answer = chain({"question": query}, return_only_outputs=True)
        result = answer['result']
    except IndexError :
        result = "No relevant data found in Documents"
        
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result)

    
    
if process_url_clicked and query:    
    vector_index = process_urls()
    get_answers(vector_index=vector_index)

