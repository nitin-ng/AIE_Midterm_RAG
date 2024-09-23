import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import QdrantVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Load documents
documents1 = PyMuPDFLoader(file_path="https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf").load()
documents2 = PyMuPDFLoader(file_path="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf").load()

# Split documents
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
documents1 = text_splitter.split_documents(documents1)
documents2 = text_splitter.split_documents(documents2)

# Initialize embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Initialize Qdrant client and vector store
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://qdrant:6333")
)
qdrant_client.create_collection(
    collection_name="ai_ethics_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
qdrant_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="ai_ethics_docs",
    embedding=embeddings
)
qdrant_vector_store.add_documents(documents1)
qdrant_vector_store.add_documents(documents2)

# Create retriever
retriever = qdrant_vector_store.as_retriever()

# Define prompt template
template = """
You are a helpful assistant. Act as an AI ethics expert and answer the question in a succinct way. 
If you cannot answer the question based on the context - you must say "I don't know".
Question:
{question}
Context:
{context}
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize LLM
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create retrieval-augmented QA chain
retrieval_augmented_qa_chain = (
    {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {
        "response": prompt | primary_qa_llm, 
        "context": itemgetter("context")
    }
)

@cl.on_chat_start
def start():
    cl.user_session.set("chain", retrieval_augmented_qa_chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    response = await chain.ainvoke({"question": message.content})
    await cl.Message(content=response["response"].content).send()

if __name__ == "__main__":
    cl.run()