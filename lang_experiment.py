import os
import json
import uuid 
import ollama
import gradio as gr
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


import faq_data as fd
import prompts as pmt

#set_llm_cache(SQLiteCache(database_path="llm_cache.db"))

OLLAMA_MODEL = "gemma:latest" 
EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "chroma_health_db"
OLLAMA_API_URL = "https://4dff1e7640fc.ngrok-free.app/"
QUERY_CACHE_VECTORS_DIR = "./chroma_db_query_cache" # For question embeddings
QUERY_CACHE_KV_PATH = "./query_cache_kv.json"     # For question -> answer mapping


class QueryCache:
    def __init__(self, embeddings, vector_dir, kv_path, score_threshold=0.25):
        self.embeddings = embeddings
        self.kv_path = kv_path
        self.score_threshold = score_threshold

        os.makedirs(vector_dir, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=vector_dir,
            embedding_function=self.embeddings,
            collection_name="query_cache",
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.kv_store = self._load_kv_store()
        
        print("Query cache enabled.")

    def _load_kv_store(self):
        if os.path.exists(self.kv_path):
            with open(self.kv_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_kv_store(self):
        with open(self.kv_path, 'w') as f:
            json.dump(self.kv_store, f, indent=2)

    def check(self, query: str):
        # Search for similar queries in the vector store
        results = self.vector_store.similarity_search_with_score(query, k=1)
        if results:
            doc, score = results[0]
            print(f"Similarity score for query: {score:.4f}")
            if score <= self.score_threshold:
                vector_id = doc.metadata.get("vector_id")
                if vector_id in self.kv_store:
                    print(f" -> Found cached response with score {score:.4f}")
                    return self.kv_store[vector_id]
        return None

    def update(self, query: str, response: str):
        # Generate a unique ID for this new entry
        vector_id = str(uuid.uuid4())
        
        # Add the query embedding to the vector store
        self.vector_store.add_texts(
            texts=[query],
            ids=[vector_id],
            metadatas=[{"vector_id": vector_id}]
        )
        
        # Add the ID and response to the key-value store
        self.kv_store[vector_id] = response
        self._save_kv_store()



llm = Ollama(model=OLLAMA_MODEL,base_url=OLLAMA_API_URL)
# Initialize the Ollama model for creating embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=OLLAMA_API_URL)

query_cache = QueryCache(
    embeddings=embeddings,
    vector_dir=QUERY_CACHE_VECTORS_DIR,
    kv_path=QUERY_CACHE_KV_PATH,
)




def get_or_create_vectorstore():
    """
    Creates a new Chroma vector store if one doesn't exist, otherwise loads the existing one.
    """
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading existing vector store...")
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings,collection_name="my_faqs")
        print("Vector store loaded successfully.")
        return vectorstore
    else:
        documents = []
        for faq_item in fd.faq_data:
            
            question = faq_item["question"]
            answer = faq_item["answer"]
            
            # Create a Document with combined content and useful metadata
            doc = Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={"source": f"faq_id_{faq_item['id']}"}
            )
            documents.append(doc)

        print(f"Creating vector store with {len(documents)} FAQ entries...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name='my_faqs'
        )
        print("Collection size:", vectorstore._collection.count())
    
        print("Vector store created and persisted.")
        return vectorstore
    
vectorstore = get_or_create_vectorstore()

def healthcare_chatbot(user_query):

    relevant_docs = vectorstore.similarity_search(user_query, k=1)

    context = "\n\n".join([doc.page_content for doc in relevant_docs]) 

    
    
    rag_prompt = pmt.RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)

    rag_prompt_2 = pmt.RAG_PROMPT_TEMPLATE_2.format(context=context, question=user_query)

    rag_prompt_3 = pmt.RAG_PROMPT_TEMPLATE_3.format(context=context, question=user_query)

    cached_response = query_cache.check(user_query)
    if cached_response:
        
        print("\n(CACHE HIT)")
        response = cached_response
    else:
    
        response = llm.invoke(rag_prompt_3)
        query_cache.update(user_query, response)

    
    return response

with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Healthcare Information Assistant Chatbot\nThis chatbot can answer general health FAQs, lifestyle tips, and clinic info.\n\n⚠️ Disclaimer: This is for educational purposes only and not a substitute for professional medical advice.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask me anything about health, lifestyle, or hospital info")
    clear = gr.Button("Clear Chat")

    def respond(message, chat_history):
        bot_reply = healthcare_chatbot(message)
        chat_history.append((message, bot_reply))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)


demo.launch()