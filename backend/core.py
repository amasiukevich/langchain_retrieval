import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.vectorstores import Pinecone
import pinecone

from typing import Any


from constants import INDEX_NAME

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def run_llm(query: str) -> Any:
    n_docs_context = 4
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    # embedded_query = embeddings.embed_query(query)
    # most_relevant = docsearch.similarity_search(k=n_docs_context)

    chat_llm = ChatOpenAI(verbose=True, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa_chain({"query": query})


if __name__ == "__main__":
    print("querying the LLM")
    result = run_llm(query="What is RetrievalQA chain?")
    print(result)
