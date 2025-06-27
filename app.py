from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

os.environ["PINECONE_API_KEY"] = "pcsk_71c9ie_7De9UmTbVsACN8Qbxgpb53UdTBphbFT2hb3wR71LF44kRnbhENewsvKwVJPr6hc"
os.environ["GROQ_API_KEY"] = "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"

embeddings = download_huggingface_embeddings()


index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
     index_name=index_name,
     embedding=embeddings
 )

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k":3})


llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response: ", response["answer"])
#     return str(response["answer"])
@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()  # Parse JSON body
    msg = data.get("message", "")  # Safe access
    print("User Input:", msg)
    
    response = rag_chain.invoke({"input": msg})  # Assuming rag_chain returns dict with 'answer'
    print("Response:", response["answer"])
    
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=True)

