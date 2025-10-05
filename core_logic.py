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
        """Sets up the vector store and retriever from the document text."""
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
        time.sleep(5)

        vector_store = PineconeLangChain.from_documents(docs, embeddings, index_name=self.index_name)
        self.retriever = vector_store.as_retriever()
        logging.info("Knowledge base and retriever are ready.")

    def _setup_llm(self):
        """Sets up the language model pipeline."""
        model_id = "distilbert-base-uncased-distilled-squad"
        qa_pipeline = pipeline("question-answering", model=model_id)
        self.llm = HuggingFacePipeline(pipeline=qa_pipeline)
        logging.info(f"LLM with model '{model_id}' is ready.")

    def _get_feedback(self):
        """Simulates feedback on a response."""
        feedback = random.choice(["good", "not helpful", "too vague"])
        logging.info(f"Simulated Feedback: {feedback}")
        return feedback

    def run(self, query: str):
        """Runs the full agentic workflow for a given query."""
        # This list will store the conversation flow for the UI
        workflow_log = []

        # 1. Get Initial Response
        logging.info(f"Processing query: {query}")
        retrieved_docs = self.retriever.invoke(query)
        if not retrieved_docs:
            initial_response = "I could not find relevant information in the document."
        else:
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            result = self.llm.pipeline(question=query, context=context)
            initial_response = result.get('answer', 'No answer could be found.')

        workflow_log.append({"type": "answer", "content": f"**Answer:** {initial_response}"})
        logging.info(f"Initial Response: {initial_response}")

        # 2. Feedback Loop
        current_response = initial_response
        for i in range(2):  # Limit to 2 iterations
            feedback = self._get_feedback()
            workflow_log.append({"type": "feedback", "content": f"Simulated Feedback: **{feedback}**"})

            if feedback == "good":
                workflow_log.append({"type": "confirmation", "content": "Feedback is good. Answer confirmed."})
                break

            # Adjust response based on feedback [cite: 10]
            if feedback == "too vague":
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                current_response = f"{current_response}\n\n**More Info:** {context[:200]}..."
            elif feedback == "not helpful":
                new_prompt = f"Please provide a more detailed and alternative answer for: {query}"
                new_docs = self.retriever.invoke(new_prompt)
                context = "\n".join([doc.page_content for doc in new_docs])
                result = self.llm.pipeline(question=new_prompt, context=context)
                current_response = result.get('answer', 'I was unable to find a better answer.')

            workflow_log.append({"type": "answer", "content": f"**Updated Answer:** {current_response}"})
            logging.info(f"Adjusted Response ({i + 1}): {current_response}")

        return workflow_log
