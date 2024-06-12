import streamlit as st
import Extract_Load as el
import Chain_Components as cc
import Langchain_Chain as lc
import os
import tempfile
#chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
os.environ["OPENAI_API_KEY"] = "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"


st.title("Agente Cognitivo")
st.sidebar.title("Configurações do Agente Cognitivo")

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.8)
OCR_flag = st.sidebar.checkbox("OCR", value=False)
extractor = st.sidebar.selectbox("Text Extractor Model", options=["textract"], index=0)
embedding_model = st.sidebar.selectbox("Embedding Model", options=["all-MiniLM-L6-v2", "other-model"], index=0)
vector_store_model = st.sidebar.selectbox("Vector Store", options=["DocArrayInMemorySearch", "other-store"], index=0)
memory_model = st.sidebar.selectbox("Memory", options=["ConversationBufferMemory", "other-memory"], index=0)
llm_model = st.sidebar.selectbox("LLM", options=["ChatOpenAI_gpt-3.5-turbo", "ChatOpenAI_gpt-4o", "ChatOpenAI_gpt-4"], index=0)
max_tokens_limit = st.sidebar.slider("Max Tokens Limit", min_value=1, max_value=5000, value=1000)
few_shot_examples = None 


prompt = st.text_input("Defina o prompt inicial: ", "")
bot_name  = st.text_input("Defina o nome do bot: ", "")
user_input = st.text_area("Digite suas perguntas, e utilize como separador de perguntas '//':", "")

st.session_state.questions = user_input.split("//")

if 'question_index' not in st.session_state:
    st.session_state.question_index = 0

if 'history' not in st.session_state:
    st.session_state.history = []

def add_message(user_message, bot_response):
    st.session_state.history.append({"user": user_message, "bot": bot_response})


match OCR_flag:
    case "True":
        OCR_flag = True
    case "False":
        OCR_flag = False
match extractor:
    case "textract":
        extractor = el.Extract().textract

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    #type=list(DocumentLoader.supported_extentions.keys()),
    accept_multiple_files=True
    )

docs = []
temp_dir = tempfile.TemporaryDirectory()
for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
        f.write(file.getvalue())
    docs.extend(el.Load(temp_filepath, extractor).load_document(OCR_flag))

splits = cc.TextSplitter(docs).RecursiveCharacterTextSplitter()
match embedding_model:
    case "all-MiniLM-L6-v2":
        embeddings = cc.Embeddings().HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    case "all-mpnet-base-v2":
        embeddings = cc.Embeddings().HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2"
        )
match vector_store_model:
    case "FAISS":
        vector_store = cc.VectorStore(splits, embeddings).FAISS()
    case "Pinecone":
        vector_store = cc.VectorStore(splits, embeddings).Pinecone()
    case "DocArrayInMemorySearch":
        vector_store = cc.VectorStore(splits, embeddings).DocArrayInMemorySearch()
retriever = cc.Retriever(embeddings, vector_store).configure_retriever(
    use_compression=False,
    similarity_threshold=0.1,
    search_type="mmr",
    k=2,
    fetch_k=4,
)
match memory_model:
    case "ConversationBufferMemory":
        memory = cc.Memory().ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    case None:
        memory = None
match llm_model:
    case "ChatOpenAI_gpt-4":
        llm = cc.LLM().ChatOpenAI(
            model_name="gpt-4", temperature=temperature, streaming=True
        )
    case "ChatOpenAI_gpt-3.5-turbo":
        llm = cc.LLM().ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=temperature, streaming=True
        )
    case "ChatOpenAI_gpt-4o":
        llm = cc.LLM().ChatOpenAI(
            model_name="gpt-4o", temperature=temperature, streaming=True
        )
user_query = st.text_input("Você: ", "")
perguntas = st.session_state.questions
i = st.session_state.question_index

if st.button("Enter"):
    try:
        part_intro = f"A questão é a introdução do entrevistado. Se introduza também, seu nome é {bot_name}"
        part_2 = f"""Você como Entrevistador deve reagir à introdução e fazer a pergunta: {perguntas[i]},
                    podendo usar o contexto para enriquecer a pergunta."""
        part_context = " {context}."
        part_answer = f"A questão é a resposta do Entrevistado para a pergunta {perguntas[i-1]} do conjunto de perguntas: {perguntas}."
        part_middle = f"""Você como Entrevistador deve reagir à resposta e fazer a pergunta: {perguntas[i]}, 
                    podendo usar o contexto para enriquecer a pergunta."""
        part_finish = f"""Você como Entrevistador deve reagir deve reagir à resposta e finalizar a entrevista,
                    podendo usar o contexto para enriquecer a pergunta.
                    """
        if i==0:
            prompt_template = part_intro+part_2+prompt+part_context
        if 0<i<len(perguntas)-1:
            prompt_template = part_answer+part_middle+prompt+part_context
        if i==len(perguntas)-1:
            prompt_template = part_answer+part_finish+prompt+part_context

        if few_shot_examples is not None:
            qa_chain = lc.Chain(
                llm, retriever, memory
            ).RetrievalChain_with_few_shot_examples(
                few_shot_examples, prompt_template, max_tokens_limit
            )
        else:
            qa_chain = lc.Chain(llm, retriever, memory).RetrievalChain_with_prompt(
                prompt_template, max_tokens_limit
            )
            bot_response = qa_chain.run(user_query)
    except NameError as e:
        st.session_state.question_index = 0
        i = st.session_state.question_index
        prompt_template = part_intro+part_2+prompt+part_context
        if few_shot_examples is not None:
            qa_chain = lc.Chain(
                llm, retriever, memory
            ).RetrievalChain_with_few_shot_examples(
                few_shot_examples, prompt_template, max_tokens_limit
            )
        else:
            qa_chain = lc.Chain(llm, retriever, memory).RetrievalChain_with_prompt(
                prompt_template, max_tokens_limit
            )
            bot_response = qa_chain.run(user_query)

    add_message(user_query, bot_response)
    st.write("### Histórico da Conversa")
    for chat in st.session_state.history:
        st.write(f"**Você 👨‍💼:** {chat['user']}")
        st.write(f"**{bot_name} 🤖:** {chat['bot']}")
st.session_state.question_index += 1


if st.sidebar.button("Limpar Histórico"):
    st.session_state.history = []
    st.session_state.question_index = 0
    st.session_state.questions = []


if st.button("Debug: Session State"):
    st.write(st.session_state)
