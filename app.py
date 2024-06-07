from flask import Flask, request, jsonify
import boto3
import os
import Extract_Load as el
import Chain_Components as cc
import Langchain_Chain as lc

app = Flask(__name__)
textract = boto3.client("textract")

# Configure API key
os.environ["OPENAI_API_KEY"] = "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"


@app.route("/upload", methods=["POST"])
def upload_files():
    print("iniciou")
    data = request.get_json()
    print("funcionou")
    config = data.get("config", {})
    temperature = float(config.get("temperature", 0.5))
    extractor = config.get("extractor")
    OCR_flag = config.get("OCR")
    embedding_model = config.get("embedding_model")
    vector_store_model = config.get("vector_store")
    memory_model = config.get("memory")
    llm_model = config.get("llm")
    max_tokens_limit = float(config.get("max_tokens_limit", 1000))

    temp_filepath = data.get("path_selected_filename")
    few_shot_examples = data.get("few_shot_examples")
    message_template = data.get("message_template")
    context_template = data.get("context_template")

    # Retornar a resposta em JSON
    response = {
        "path_selected_filename": temp_filepath,
        "few_shot_examples": few_shot_examples,
        "message_template": message_template,
        "context_template": context_template,
    }
    match OCR_flag:
        case "True":
            OCR_flag = True
        case "False":
            OCR_flag = False
    match extractor:
        case "textract":
            extractor = el.Extract().textract

    docs = el.Load(temp_filepath, extractor).load_document(OCR_flag)
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
        case "ChatOpenAI":
            llm = cc.LLM().ChatOpenAI(
                model_name="gpt-4", temperature=temperature, streaming=True
            )
    if few_shot_examples is not None:
        qa_chain = lc.Chain(
            llm, retriever, memory
        ).RetrievalChain_with_few_shot_examples(
            few_shot_examples, message_template, context_template, max_tokens_limit
        )
    else:
        qa_chain = lc.Chain(llm, retriever, memory).RetrievalChain_with_prompt(
            message_template, context_template, max_tokens_limit
        )
    app.config["qa_chain"] = qa_chain
    return jsonify(response)


@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("message", "")
    if not user_query:
        return jsonify({"error": "Mensagem não fornecida"}), 400
    qa_chain = app.config.get("qa_chain")
    if not qa_chain:
        return jsonify({"error": "QA chain não configurada"}), 400
    response = qa_chain.run(user_query)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()
