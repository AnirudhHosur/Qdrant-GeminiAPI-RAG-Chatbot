import os
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Set up the embeddings
MODEL = 'paraphrase-MiniLM-L6-v2'
embeddings = SentenceTransformer(MODEL)

# Load PDF documents from the data folder
data_folder = "data"
documents = []
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, filename)
        loader = PyPDFLoader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        documents.extend(loader.load_and_split(text_splitter))

# Set up Qdrant client
client = QdrantClient("localhost", port=6333)

# Create collection in Qdrant database
if not client.collection_exists("document_collection"):
    client.create_collection(
        collection_name="document_collection",
        vectors_config={
            "content": VectorParams(size=384, distance=Distance.COSINE)
        }
    )

# Function to chunk documents and upload to Qdrant
def chunked_metadata(data, client=client, collection_name="document_collection", batch_size=100):
    chunked_metadata = []

    for i, item in enumerate(data):
        id = str(uuid4())
        content = item.page_content
        source = item.metadata["source"]
        page = item.metadata["page"]

        content_vector = embeddings.encode([content])[0]
        vector_dict = {"content": content_vector}

        payload = {
           "page_content": content,
           "metadata": {
                "id": id,
                "page_content": content,
                "source": source,
                "page": page,
           }
        }

        metadata = PointStruct(id=id, vector=vector_dict, payload=payload)
        chunked_metadata.append(metadata)

        # Upload in batches
        if len(chunked_metadata) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=chunked_metadata
            )
            chunked_metadata = []  # Reset after each batch

    # Upload any remaining points
    if chunked_metadata:
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=chunked_metadata
        )


# Upload all documents to Qdrant
chunked_metadata(documents)

# Print information about the collection
document_collection = client.get_collection("document_collection")
print(f"Points in collection: {document_collection.points_count}")