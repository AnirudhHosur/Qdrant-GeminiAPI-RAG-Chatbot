from sentence_transformers import SentenceTransformer
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from claude import ClaudeLLM
from gemini import GeminiLLM

# Set up the embeddings
embeddings = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embedding_function(text):
    if isinstance(text, str):
        text = [text]
    return embeddings.encode(text)[0].tolist()

# Set up Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# Set up a vector store using our Qdrant collection
vectorstore = Qdrant(client=qdrant_client,
                     collection_name="document_collection",
                     embeddings=embedding_function,
                     vector_name="content")

# Set up a retriever to fetch relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define a template for our chatbot's responses
template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

Context: {context}

Question: {question}

Answer:
"""

# Initialize the LLMs
claude_llm = ClaudeLLM()
gemini_llm = GeminiLLM()

def get_response(question, llm_choice):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Prepare context from retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the prompt with context and question
    prompt = template.format(context=context, question=question)
    
    # Get response from the chosen LLM
    if llm_choice.lower() == 'claude':
        response = claude_llm.generate_response(prompt)
    elif llm_choice.lower() == 'gemini':
        response = gemini_llm.generate_response(prompt)
    else:
        response = "Invalid LLM choice. Please choose either 'claude' or 'gemini'."
    
    return context, response

# Example usage
if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        llm_choice = input("Choose LLM (claude/gemini): ")
        
        context, response = get_response(question, llm_choice)
        print(f"\nQuestion: {question}")
        print(f"Context: {context}")
        print(f"Answer ({llm_choice}): {response}\n")