from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts.prompt import PromptTemplate
import logging
import pathlib
from langchain.schema import Document
from typing import Any
from langchain.document_loaders import (
 PyPDFLoader, TextLoader,
 UnstructuredWordDocumentLoader,
 UnstructuredEPubLoader
)

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from sentence_transformers import SentenceTransformer
import os
import tempfile
import boto3

textract = boto3.client('textract')


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], ** kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", 
        strategy="fast")

class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    """Loads in a document with a supported extension."""
    supported_extentions = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".epub": EpubReader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader
    }

def sort_key_with_tolerance(tolerance=0.01):
    def key(box):
        top = box['bbox']['Top']
        left = box['bbox']['Left']
        return (round(top / tolerance), left)
    return key


def extract_and_organize_text(image_bytes, vertical_diff=0.1):
    response = textract.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=["TABLES", "FORMS"])
    
    blocks = response['Blocks']
    lines = []
    for block in blocks:
        if block['BlockType'] == 'LINE':
            lines.append({
                'text': block['Text'],
                'bbox': block['Geometry']['BoundingBox']
            })
    
    #lines = sorted(lines, key=lambda x: (x['bbox']['Top'], x['bbox']['Left']))
    lines = sorted(lines, key=sort_key_with_tolerance())
    organized_text = []
    current_top = -1

    for line in lines:
        if current_top == -1 or abs(line['bbox']['Top'] - current_top) > vertical_diff:
            organized_text.append(line['text'])
            current_top = line['bbox']['Top']
        else:
            organized_text[-1] += ' ' + line['text']
    
    extracted_text = '___'.join(organized_text)
    return extracted_text


def load_document(temp_filepath: str, OCR) -> list[Document]:
    """Load a file and return it as a list of documents."""
    if OCR:
        with open(temp_filepath, 'rb') as image_file:
            image_bytes = image_file.read()
            extracted_text = extract_and_organize_text(image_bytes)
            st.write(extracted_text)
        docs = [Document(page_content=extracted_text)]
    else:
        ext = pathlib.Path(temp_filepath).suffix
        loader = DocumentLoader.supported_extentions.get(ext)
        if not loader:
            raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
            )
        if loader == TextLoader:
            loader = loader(temp_filepath, encoding = 'UTF-8')
        else:
            loader = loader(temp_filepath)
        docs = loader.load()
    logging.info(docs)
    return docs


def configure_retriever(docs: list[Document], use_compression=False) -> BaseRetriever:
    """Retriever to use."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )
    if not use_compression:
        return retriever
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.1
    )
    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )


def configure_chain(retriever: BaseRetriever, temp, mood, few_shots=False, n_tokens = 4000) -> Chain:
    """Configure chain with a retriever."""
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Setup LLM and QA chain; set temperature low to keep hallucinations in check
    llm = ChatOpenAI(
    model_name="gpt-4", temperature=temp, streaming=True
    )
    # Passing in a max_tokens_limit amount automatically
    # truncates the tokens when prompting your llm!
    if mood =='Custom':
        message_template = 'Perguntas serão separadas por //.Seja direto. Responda somente o necessário. Qualquer data deve ser fornecida no formato MM-DD-YYYY. A respota dever ser um json onde key é a pargunta e value a respota'
    else:
        message_template = f"Be {mood}."


    context_template = r"""
        {context}
        O número de inscrição tem 12 dígitos.
        O DV tem dois digitos, os dois últimos digitos do número de incrição de 12 dígitos
        """

    general_system_template = message_template+context_template
    general_user_template = "Question:```{question}```"
    if few_shots:
        few_shot_examples = [
        {"input":"Qual o nome?",
        "output":"Bruno Borges de Souza"},
        ]

        few_shot_template = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}")
            ]
        )

        messages = [
            FewShotChatMessagePromptTemplate(example_prompt=few_shot_template, examples=few_shot_examples),
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
    else:
        messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=n_tokens, combine_docs_chain_kwargs={'prompt': qa_prompt}
    )


def configure_qa_chain(uploaded_files, path_selected_filename, temp=0, mood="Formal", n_tokens = 4000, OCR=False):
    """Read documents, configure retriever, and the chain."""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in path_selected_filename:
        docs.extend(load_document(file, OCR))

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath, OCR))
    retriever = configure_retriever(docs=docs, use_compression=True)
    return configure_chain(retriever=retriever, temp = temp, mood = mood, n_tokens = n_tokens)


if __name__ == "__main__":
    st.set_page_config(page_title="LangChain: Chat with Documents", page_icon=" ")
    st.title(" LangChain: Chat with Documents")
    textract_flag = st.checkbox("Activate OCR with AWS Textract")
    uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    #type=list(DocumentLoader.supported_extentions.keys()),
    accept_multiple_files=True
    ) 


    temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    tokens = st.sidebar.slider('Number maximum of tokens', min_value=100, max_value=10000, value=4000, step=50)
    tones_of_conversation = [
    "Custom",
    "Formal 👔",
    "Informal 😊",
    "Friendly 👋",
    "Professional 💼",
    "Serious ⚠️",
    "Humorous 😄",
    "Direct ➡️",
    "Empathetic ❤️",
    "Polite 🌸",
    "Respectful 🙏",
    "Ironic 🤔"
]
    tones_of_conversation_with_captions = [
    "Custom template"
    "Suitable for official or professional contexts.",
    "Casual and relaxed style, typically used among friends or acquaintances.",
    "Warm and welcoming, aimed at fostering a positive relationship.",
    "Conveys competence and expertise, often used in business settings.",
    "Conveys gravity and importance, suitable for critical or sensitive matters.",
    "Includes elements of humor or playfulness, lightens the mood.",
    "Clear and to the point, avoiding unnecessary elaboration.",
    "Shows understanding and compassion, acknowledging the emotions of others.",
    "Marked by courtesy and consideration for others, often includes pleasantries.",
    "Shows esteem and regard for the other party, maintaining dignity and decorum.",
    "Uses irony or sarcasm to convey a different meaning than what is explicitly stated."
    ]   
    mood = st.sidebar.radio('Tone of conversation',tones_of_conversation)

    folder_path='./books'
    filenames = os.listdir(folder_path)
    selected_filename = st.multiselect('Select a file', filenames)
    path_selected_filename = [os.path.join(folder_path, file) for file in selected_filename]
    if not uploaded_files and not path_selected_filename:
        st.info("Please upload documents to continue.")
        st.stop()

    qa_chain = configure_qa_chain(uploaded_files, path_selected_filename, temp=temperature, mood=mood, n_tokens=tokens, OCR=textract_flag)
    assistant = st.chat_message("assistant")
    user_query = st.chat_input(placeholder="Ask me anything!")
    if user_query:
        stream_handler = StreamlitCallbackHandler(assistant)
        response = qa_chain.run(user_query, callbacks=[stream_handler])
        st.markdown(response)
