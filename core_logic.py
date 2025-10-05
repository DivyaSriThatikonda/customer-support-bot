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
        logging.info("Initializing SupportBotAgent...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "support-bot-index"
        self._setup_knowledge_base(document_text)
        self._setup_llm()

    def _setup_knowledge_base(self, document_text: str):
        logging.info("Setting up knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.create_documents([document_text])
        
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # reset index if already exists
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

        # ✅ Fetch top 3 docs
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        logging.info("Knowledge base and retriever are ready.")

    def _setup_llm(self):
        # ✅ Switch to Flan-T5 for natural generation
        model_id = "google/flan-t5-base"
        gen_pipeline = pipeline("text2text-generation", model=model_id, max_length=256)
        self.llm = HuggingFacePipeline(pipeline=gen_pipeline)
        logging.info(f"LLM with model '{model_id}' is ready.")

    def _get_feedback(self):
        feedback = random.choice(["good", "not helpful", "too vague"])
        logging.info(f"Simulated Feedback: {feedback}")
        return feedback

    def _generate_answer(self, query, context, style="normal"):
        """Helper to generate clean answers with Flan-T5."""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer clearly and helpfully."
        if style == "detailed":
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nGive a more detailed explanation."
        elif style == "rephrase":
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nExplain the answer differently."
        
        result = self.llm.pipeline(prompt)
        return result[0]['generated_text']

    def run(self, query: str):
        """Runs the full agentic workflow for a given query."""
        workflow_log = []

        logging.info(f"Processing query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            initial_response = "I could not find relevant information in the document."
        else:
            context = " ".join([doc.page_content[:400] for doc in retrieved_docs])
            initial_response = self._generate_answer(query, context)

        workflow_log.append({"type": "answer", "content": f"**Answer:** {initial_response}"})
        current_response = initial_response

        # ✅ Feedback loop with generative refinement
        for _ in range(2):
            feedback = self._get_feedback()
            workflow_log.append({"type": "feedback", "content": f"Simulated Feedback: **{feedback}**"})

            if feedback == "good":
                workflow_log.append({"type": "confirmation", "content": "Feedback is good. Answer confirmed."})
                break

            elif feedback == "too vague":
                more_docs = self.retriever.invoke(query)
                more_context = " ".join([doc.page_content[:400] for doc in more_docs])
                current_response = self._generate_answer(query, more_context, style="detailed")

            elif feedback == "not helpful":
                current_response = self._generate_answer(query, context, style="rephrase")

            workflow_log.append({"type": "answer", "content": f"**Updated Answer:** {current_response}"})

        return workflow_log
