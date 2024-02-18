from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough



def read_pdf():
    reader = PdfReader("Resume_E_Abhishek_Dec_15.pdf")
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text()
    print(page)
    print(type(page))
    print(number_of_pages)
    with open("file.txt", "w") as f:
        f.write(text)


def split_text():
    with open("file.txt", "r") as f:
        contents = f.read()

    print(len(contents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len,
                                                   is_separator_regex=False, separators=["."])
    split_docs = text_splitter.split_text(contents)

    with open("file2.txt","w") as f:
        for doc in split_docs:
            f.write(doc)
            f.write('\n\n')
    vectorstore = Chroma.from_documents(
        documents=split_docs, embedding=OllamaEmbeddings(model="llama2"))


def get_promt():
    prompt = hub.pull('rlm/rag-prompt')


if __name__ == "__main__":
    # read_pdf()
    split_text()
