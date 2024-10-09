import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
import chainlit as cl

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient("http://localhost:6333")
    print("Successfully connected to Qdrant")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    raise

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

# Create or get existing collection
COLLECTION_NAME = "ai_ethics_docs"
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} already exists")
except Exception:
    print(f"Creating new collection {COLLECTION_NAME}")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Initialize vector store
qdrant_vector_store = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)

# Add documents to vector store
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
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

@cl.on_chat_start
async def start():
    cl.user_session.set("chain", retrieval_augmented_qa_chain)

    welcome_message = """
    Welcome to the AI Ethics Expert Assistant!

    This system is based on two important documents:
    1. The Blueprint for an AI Bill of Rights
    2. NIST AI Risk Management Framework

    You can ask questions about AI ethics, rights, and risk management based on these documents. 
    The system will provide answers using the content from these sources.

    Here are some example questions you could ask:
    - What are the key principles in the AI Bill of Rights?
    - How does the NIST framework approach AI risk management?
    - What safeguards are recommended for AI systems?
    - How should AI systems protect user privacy?

    Feel free to ask any question related to AI ethics and governance. If the answer isn't in the documents, 
    the system will let you know.

    Let's get started! What would you like to know about AI ethics and governance?
    """

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    response = await chain.ainvoke({"question": message.content})
    await cl.Message(content=response["response"].content).send()

if __name__ == "__main__":
    import chainlit as cl
    from chainlit.cli import run_chainlit
    import sys

    # Check if the script is being run directly
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        cl.run()
    else:
        # If not, use run_chainlit to start the Chainlit server
        run_chainlit("midterm_hfrag_chainlit.py")