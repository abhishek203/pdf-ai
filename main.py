from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

def read_pdf():
    reader = PdfReader("Resume_E_Abhishek_Dec_15.pdf")
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()
    print(page)
    print(type(page))
    print(number_of_pages)
    with open("file.txt","a") as f:
        f.write(text)

def split_text():
    with open("file.txt","r") as f:
        contents = f.read()
    
    print(len(contents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function = len,
    is_separator_regex = False)
    split_docs = text_splitter.create_documents([contents])
    
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=OllamaEmbeddings(model="llama2"))

if __name__ == "__main__":
    split_text()