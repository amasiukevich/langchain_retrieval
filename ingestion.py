import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import pinecone

from constants import INDEX_NAME

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/python.langchain.com/en/latest", features="lxml"
    )
    raw_documents = loader.load()
    print(f"Hey there. Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)

    print(f"Splitted into {len(documents)} chunks")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )

    print("*************** Persisted the vectors into the vector store ***************")


if __name__ == "__main__":
    ingest_docs()
