from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai  import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

from langchain.globals import set_debug
import os
from dotenv import load_dotenv

set_debug(True)
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

loaded_text = [
  PyPDFLoader("./files/GTB_gold_Nov23.pdf"),
  PyPDFLoader("./files/GTB_black_Nov23.pdf"),
  PyPDFLoader("./files/GTB_platinum_Nov23.pdf"),
  PyPDFLoader("./files/GTB_standard_Nov23.pdf"),
  ]

documents = []

for loader in loaded_text:
    # Load each PDF file and extend the documents list
    documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

question = "Como devo proceder caso tenho um item comprado roubado?"
result = qa_chain.invoke({"query" : question})
print(result)