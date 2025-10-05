import streamlit as st
from core_logic import SupportBotAgent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

# --- Page Config & CSS ---
st.set_page_config(page_title="Agentic Support Bot", page_icon="🤖", layout="wide")
st.markdown("""
<style>
/* (Your custom CSS) */
body { background-color: #FAF3E0; }
.header-box { background: #FFFFFF; border: 1px solid #E5DCC3; padding: 18px; border-radius: 12px; text-align: center; font-size: 24px; font-weight: 600; color: #4E423D; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
.chat-bubble { padding: 14px 20px; border-radius: 18px; margin: 6px 0; max-width: 80%; clear: both; line-height: 1.6; }
.user-bubble { background-color: #FAB488; color: #4E423D; border-radius: 18px 18px 4px 18px; float: right; border: 1px solid #F8A56E; }
.assistant-bubble { background-color: #FFFFFF; color: #4E423D; border: 1px solid #E5DCC3; border-radius: 18px 18px 18px 4px; float: left; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- Helper Function ---
def extract_text_from_file(uploaded_file):
    # ... (This function remains the same)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path) if uploaded_file.type == "application/pdf" else TextLoader(temp_file_path)
        text = "\n".join([doc.page_content for doc in loader.load()])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        text = ""
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return text

# --- Session State Initialization ---
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if st.button("🚀 Process Document"):
        if uploaded_file:
            with st.spinner("Processing document... This may take a moment."):
                document_text = extract_text_from_file(uploaded_file)
                if document_text:
                    # --- THE FIX IS HERE ---
                    # Create a new agent directly instead of using a cached function.
                    st.session_state.agent = SupportBotAgent(document_text=document_text)
                    st.session_state.messages = [] # Clear history for the new document
                    st.success("✅ Document processed! Ready to chat.")
        else:
            st.error("Please upload a document first.")

# --- Main App Area ---
st.markdown("<div class='header-box'>🤖 Agentic Customer Support Bot</div>", unsafe_allow_html=True)

# Display chat messages and handle input
if st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f"<div class='chat-bubble user-bubble'>{content}</div>", unsafe_allow_html=True)
        elif role == "answer":
            st.markdown(f"<div class='chat-bubble assistant-bubble'>{content}</div>", unsafe_allow_html=True)
        elif role == "feedback":
            st.info(content)
        elif role == "confirmation":
            st.success(content)

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking and iterating..."):
            workflow_log = st.session_state.agent.run(prompt)
            for item in workflow_log:
                st.session_state.messages.append({"role": item["type"], "content": item["content"]})
        st.rerun()
else:
    st.info("Please upload a document and click 'Process Document' in the sidebar to begin.")
