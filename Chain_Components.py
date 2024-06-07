from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores import DocArrayInMemorySearch, FAISS, Pinecone
from langchain.schema import Document
from typing import List


class TextSplitter:
    def __init__(self, docs: List[Document]):
        self.docs = docs

    def RecursiveCharacterTextSplitter(
        self, chunk_size=1000, chunk_overlap=300
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(self.docs)
        return splits


class Embeddings:
    def HuggingFaceEmbeddings(self, model_name: str) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings


class VectorStore:
    def __init__(
        self, splits: List[Document], embeddings: HuggingFaceEmbeddings
    ) -> None:
        self.splits = splits
        self.embeddings = embeddings

    def DocArrayInMemorySearch(self) -> DocArrayInMemorySearch:
        return DocArrayInMemorySearch.from_documents(self.splits, self.embeddings)

    def FAISS(self) -> FAISS:
        return FAISS.from_documents(self.splits, self.embeddings)

    def Pinecone(self) -> Pinecone:
        return Pinecone(self.splits, self.embeddings)


class Retriever:
    def __init__(self, embeddings, vector_store):
        self.embeddings = embeddings
        self.vector_store = vector_store

    def configure_retriever(
        self,
        use_compression=False,
        similarity_threshold=0.1,
        search_type="mmr",
        k=2,
        fetch_k=4,
    ):
        retriever = self.vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k}
        )
        if not use_compression:
            return retriever
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings, similarity_threshold=similarity_threshold
        )
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=retriever
        )


class Memory:
    def ConversationBufferMemory(
        self, memory_key: str = "chat_history", return_messages: bool = True
    ):
        return ConversationBufferMemory(
            memory_key=memory_key, return_messages=return_messages
        )


class LLM:
    def ChatOpenAI(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        streaming: bool = True,
    ):
        return ChatOpenAI(
            model_name=model_name, temperature=temperature, streaming=streaming
        )
