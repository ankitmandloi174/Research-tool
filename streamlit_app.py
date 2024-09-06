import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import GooglePalmEmbeddings,OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
import conf

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("ResearchBot")
st.sidebar.title("OpenAI key")

OPENAI_API_KEY = st.sidebar.text_input(f"OpenAI key")

if OPENAI_API_KEY:

    st.sidebar.title("Article URLs")

    import os
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    llm = ChatOpenAI(model="gpt-4o-mini")
    main_placeholder = st.empty()

    urls = []

    for i in range(2):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    query = st.text_input("Question: ")

    query_button_clicked = st.button("Find Answer")
    process_url_clicked = st.sidebar.button("Process URLs")


    def process_urls():
        
        main_placeholder.text("Data Loading...Started...✅✅✅")
        
        loader  = UnstructuredURLLoader(urls= urls)
        data = loader.load()
        
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        
        spliiter  = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=1000, chunk_overlap=0)
        docs = spliiter.split_documents(data)
        
        vectordb = Chroma.from_documents(documents=docs, embedding = OpenAIEmbeddings(),persist_directory='db')
    
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)
        vectordb.persist()
        main_placeholder.text("Embedding Generated Successfully...✅✅✅")


    if process_url_clicked:
        process_urls()

        
    def get_answers(vectorstore):
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
        chain = RetrievalQA.from_chain_type(llm=llm, 
                    chain_type="map_reduce", 
                    retriever=retriever, 
                    input_key="question", return_source_documents=True)
        import langchain
        langchain.debug = True
        try:
            answer = chain({"question": query}, return_only_outputs=True)
        
            result = answer['result']
            first_document = answer['source_documents'][1]

            # Extract the metadata from the first document
            metadata = first_document.metadata
            source_documents = metadata['source']
            
            if answer.get('source_documents',None) is None:
                result = "No relevant data found in Documents"
                
        except IndexError :
            result = "No relevant data found in Documents"
            
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result)
        st.header("Source")
        st.write(source_documents)

        
    if query_button_clicked:
        vectordb = Chroma(persist_directory="db", 
                    embedding_function=OpenAIEmbeddings())
        get_answers(vectorstore=vectordb)
        

