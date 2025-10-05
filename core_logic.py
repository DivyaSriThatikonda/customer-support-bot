import logging
import os
import random
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

load_dotenv()

logging.basicConfig(
    filename='support_bot_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SupportBotAgent:
    def __init__(self, document_text: str):
        # ... (this part remains the same)
        logging.info("Initializing SupportBotAgent...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "support-bot-index"
        self._setup_knowledge_base(document_text)
        self._setup_llm()

    def _setup_knowledge_base(self, document_text: str):
        # ... (this part remains the same)
        logging.info("Setting up knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.create_documents([document_text])
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
        self.pc.create_index(
            name=self.index_name, metric="cosine", dimension=384,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        vector_store = PineconeLangChain.from_documents(docs, embeddings, index_name=self.index_name)
        index = self.pc.Index(self.index_name)
        while True:
            stats = index.describe_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                break
            time.sleep(1)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        logging.info("Knowledge base and retriever are ready.")

    def _setup_llm(self):
        # ... (this part remains the same)
        model_id = "distilbert-base-uncased-distilled-squad"
        qa_pipeline = pipeline("question-answering", model=model_id)
        self.llm = HuggingFacePipeline(pipeline=qa_pipeline)
        logging.info(f"LLM with model '{model_id}' is ready.")

    def _get_feedback(self):
        # ... (this part remains the same)
        feedback = random.choice(["good", "not helpful", "too vague"])
        logging.info(f"Simulated Feedback: {feedback}")
        return feedback

    def run(self, query: str):
        """Runs the full agentic workflow for a given query."""
        workflow_log = []

        logging.info(f"Processing query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        # --- NEW DEBUG PRINT STATEMENT ---
        print("\n" + "="*50)
        print(f"DEBUG: For query '{query}', I retrieved the following documents:")
        if not retrieved_docs:
            print("--- NO DOCUMENTS RETRIEVED ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- DOC {i+1} ---\n{doc.page_content}\n" + "-"*15)
        print("="*50 + "\n")
        # --- END DEBUG PRINT ---

        if not retrieved_docs:
            initial_response = "I could not find relevant information in the document."
        else:
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            result = self.llm.pipeline(question=query, context=context)
            initial_response = result.get('answer', 'No answer could be found.')
        
        # ... (The rest of the feedback loop remains the same)
        workflow_log.append({"type": "answer", "content": f"**Answer:** {initial_response}"})
        current_response = initial_response
        for i in range(2):
            feedback = self._get_feedback()
            workflow_log.append({"type": "feedback", "content": f"Simulated Feedback: **{feedback}**"})
            if feedback == "good":
                workflow_log.append({"type": "confirmation", "content": "Feedback is good. Answer confirmed."})
                break
            if feedback == "too vague":
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                current_response = f"{current_response}\n\n**More Info:** {context[:200]}..."
            elif feedback == "not helpful":
                if retrieved_docs:
                    full_context = retrieved_docs[0].page_content
                    current_response = (f"I see the previous answer was not helpful. Here is the full context from the most relevant section:\n\n---\n\n> {full_context}")
                else:
                    current_response = "I was unable to find an alternative answer."
            workflow_log.append({"type": "answer", "content": f"**Updated Answer:** {current_response}"})
        return workflow_log
