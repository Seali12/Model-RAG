import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# cargar enviroment
load_dotenv() 

# conecto groq
groq_api_key = os.environ['GROQ_API_KEY']

# elijo el modelos
llm_groq = ChatGroq(
            groq_api_key=groq_api_key, model_name="llama3-70b-8192",
                         temperature=0.2)


@cl.on_chat_start
async def on_chat_start():
    files = None 

    
    while files is None:
        files = await cl.AskFileMessage(
            content="Adjunte un pdf para empezar!",
            accept=["application/pdf"],
            max_size_mb=100,# limite del archivo -> probar con mas 
            max_files=10,
            timeout=180, #tiempo de timeout
        ).send()

    texts = []
    metadatas = []
    for file in files:
        print(file) 


        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # creo la metada data para cada chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # creo el chroma  DB Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # hitorial
    message_history = ChatMessageHistory()
    
    # contexto
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    elements = [
    cl.Image(name="image", display="inline", path="pic.jpg")
    ]
    # decirle al cliente q termino de procesar lso archivos
    msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!",elements=elements)
    await msg.send()

    #guardar la session -> no funciona ver por q
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
     
    chain = cl.user_session.get("chain") 
    
    cb = cl.AsyncLangchainCallbackHandler()
    
  
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] 
    
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()