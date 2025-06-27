from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

os.environ["PINECONE_API_KEY"] = "pcsk_71c9ie_7De9UmTbVsACN8Qbxgpb53UdTBphbFT2hb3wR71LF44kRnbhENewsvKwVJPr6hc"

extracted_data=load_pdf_file("data/")
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_embeddings()

pc = Pinecone(api_key="pcsk_71c9ie_7De9UmTbVsACN8Qbxgpb53UdTBphbFT2hb3wR71LF44kRnbhENewsvKwVJPr6hc")

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)