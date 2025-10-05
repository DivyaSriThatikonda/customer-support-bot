# import logging
# import os
# import random
# import time
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from langchain_pinecone import Pinecone as PineconeLangChain
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from transformers import pipeline

# load_dotenv()

# logging.basicConfig(
#     filename='support_bot_log.txt',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# class SupportBotAgent:
#     def __init__(self, document_text: str):
#         logging.info("Initializing SupportBotAgent...")
#         self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         self.index_name = "support-bot-index"
#         self._setup_knowledge_base(document_text)
#         self._setup_llm()

#     def _setup_knowledge_base(self, document_text: str):
#         logging.info("Setting up knowledge base...")
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         docs = text_splitter.create_documents([document_text])
        
#         model_name = "all-MiniLM-L6-v2"
#         embeddings = HuggingFaceEmbeddings(model_name=model_name)

#         # reset index if already exists
#         if self.index_name in self.pc.list_indexes().names():
#             self.pc.delete_index(self.index_name)

#         self.pc.create_index(
#             name=self.index_name, metric="cosine", dimension=384,
#             spec=ServerlessSpec(cloud='aws', region='us-east-1')
#         )
#         while not self.pc.describe_index(self.index_name).status['ready']:
#             time.sleep(1)

#         vector_store = PineconeLangChain.from_documents(docs, embeddings, index_name=self.index_name)
#         index = self.pc.Index(self.index_name)

#         while True:
#             stats = index.describe_index_stats()
#             if stats.get('total_vector_count', 0) > 0:
#                 break
#             time.sleep(1)

#         # ✅ Fetch top 3 docs
#         self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#         logging.info("Knowledge base and retriever are ready.")

#     def _setup_llm(self):
#         # ✅ Switch to Flan-T5 for natural generation
#         model_id = "google/flan-t5-base"
#         gen_pipeline = pipeline("text2text-generation", model=model_id, max_length=256)
#         self.llm = HuggingFacePipeline(pipeline=gen_pipeline)
#         logging.info(f"LLM with model '{model_id}' is ready.")

#     def _get_feedback(self):
#         feedback = random.choice(["good", "not helpful", "too vague"])
#         logging.info(f"Simulated Feedback: {feedback}")
#         return feedback

#     def _generate_answer(self, query, context, style="normal"):
#         """Helper to generate clean answers with Flan-T5."""
#         prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer clearly and helpfully."
#         if style == "detailed":
#             prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nGive a more detailed explanation."
#         elif style == "rephrase":
#             prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nExplain the answer differently."
        
#         result = self.llm.pipeline(prompt)
#         return result[0]['generated_text']

#     def run(self, query: str):
#         """Runs the full agentic workflow for a given query."""
#         workflow_log = []

#         logging.info(f"Processing query: {query}")
#         retrieved_docs = self.retriever.invoke(query)

#         if not retrieved_docs:
#             initial_response = "I could not find relevant information in the document."
#         else:
#             context = " ".join([doc.page_content[:400] for doc in retrieved_docs])
#             initial_response = self._generate_answer(query, context)

#         workflow_log.append({"type": "answer", "content": f"**Answer:** {initial_response}"})
#         current_response = initial_response

#         # ✅ Feedback loop with generative refinement
#         for _ in range(2):
#             feedback = self._get_feedback()
#             workflow_log.append({"type": "feedback", "content": f"Simulated Feedback: **{feedback}**"})

#             if feedback == "good":
#                 workflow_log.append({"type": "confirmation", "content": "Feedback is good. Answer confirmed."})
#                 break

#             elif feedback == "too vague":
#                 more_docs = self.retriever.invoke(query)
#                 more_context = " ".join([doc.page_content[:400] for doc in more_docs])
#                 current_response = self._generate_answer(query, more_context, style="detailed")

#             elif feedback == "not helpful":
#                 current_response = self._generate_answer(query, context, style="rephrase")

#             workflow_log.append({"type": "answer", "content": f"**Updated Answer:** {current_response}"})

#         return workflow_log


import logging
import os
import random
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
        self._setup_llms()

    def _setup_knowledge_base(self, document_text: str):
        """Sets up the vector store and retriever from the document text."""
        logging.info("Setting up knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.create_documents([document_text])
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name, metric="cosine", dimension=384,
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True)
        logging.info("Cleared all previous vectors from the index.")
        
        vector_store = PineconeLangChain.from_documents(docs, embeddings, index_name=self.index_name)
        
        # Wait until the vector count is greater than 0
        while True:
            try:
                stats = index.describe_index_stats()
                if stats.get('total_vector_count', 0) > 0:
                    logging.info(f"Index is ready with {stats['total_vector_count']} vectors.")
                    break
            except Exception as e:
                logging.warning(f"Waiting for index stats... ({e})")
            time.sleep(1)

        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        logging.info("Knowledge base and retriever are ready.")

    def _setup_llms(self):
        """Sets up both the main generation LLM and the smaller critic LLM."""
        # Main model for generating answers
        main_model_id = "google/flan-t5-base"
        main_pipeline = pipeline("text2text-generation", model=main_model_id, max_length=256)
        self.llm = HuggingFacePipeline(pipeline=main_pipeline)
        logging.info(f"Main LLM with model '{main_model_id}' is ready.")

        # Critic model for evaluating answers
        critic_model_id = "google/flan-t5-small"
        critic_pipeline = pipeline("text2text-generation", model=critic_model_id, max_length=10)
        self.critic_llm = HuggingFacePipeline(pipeline=critic_pipeline)
        logging.info(f"Critic LLM with model '{critic_model_id}' is ready.")

    def _get_llm_feedback(self, query: str, context: str, response: str):
        """Uses the critic LLM to generate intelligent feedback."""
        prompt_template = f"""You are an AI evaluator. Based on the CONTEXT and QUESTION, evaluate the ANSWER.
        Respond with only one word: "good", "too vague", or "not helpful".
        - "good": The answer is correct and complete.
        - "too vague": The answer is correct but too short or missing details.
        - "not helpful": The answer is wrong or irrelevant.

        CONTEXT: {context[:500]}...
        QUESTION: {query}
        ANSWER: {response}
        EVALUATION:"""
        
        evaluation = self.critic_llm.invoke(prompt_template).strip().lower()
        
        if evaluation not in ["good", "too vague", "not helpful"]:
            return "good" # Default to good on invalid response
        
        logging.info(f"Simulated Feedback (LLM-Based): {evaluation}")
        return evaluation

    def _generate_llm_answer(self, query: str, context: str):
        """Generates an answer using the strict prompt template."""
        template = """
        **ROLE:** You are a hyper-focused information extraction engine.
        **TASK:** Your sole purpose is to find and extract the answer to the QUESTION from the provided CONTEXT.
        **RULES:**
        1. You must analyze the CONTEXT meticulously. Your answer must be extracted directly from this text and nothing else.
        2. Your response must be the answer itself, direct and concise.
        3. **DO NOT** provide any explanations, interpretations, summaries, or conversational filler.
        4. If the answer is not in the CONTEXT, you **MUST** respond with the exact phrase: "I'm sorry, I couldn't find the answer in the provided documents."
        ---
        **CONTEXT:**
        {context}
        ---
        **QUESTION:**
        {question}
        ---
        **ANSWER:**
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    def run(self, query: str):
        """Runs the full agentic workflow."""
        workflow_log = []
        logging.info(f"Processing query: {query}")
        
        retrieved_docs = self.retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        if not retrieved_docs:
            initial_response = "I could not find relevant information in the document."
        else:
            initial_response = self._generate_llm_answer(query=query, context=context)
        
        workflow_log.append({"type": "answer", "content": f"**Answer:** {initial_response}"})

        current_response = initial_response
        for i in range(2):
            feedback = self._get_llm_feedback(query=query, context=context, response=current_response)
            workflow_log.append({"type": "feedback", "content": f"Simulated Feedback: **{feedback}**"})
            
            if feedback == "good":
                workflow_log.append({"type": "confirmation", "content": "Feedback is good. Answer confirmed."})
                break
            
            if feedback == "too vague" and retrieved_docs:
                current_response = f"{current_response}\n\n**More Info:** {context[:200]}..."
            elif feedback == "not helpful" and retrieved_docs:
                full_context = retrieved_docs[0].page_content
                current_response = (f"I see the previous answer was not helpful. Here is the full context from the most relevant section:\n\n---\n\n> {full_context}")

            workflow_log.append({"type": "answer", "content": f"**Updated Answer:** {current_response}"})
            
        return workflow_log
