import html
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 4
USER_AVATAR = "🧑‍🎓"
ASSISTANT_AVATAR = "🤖"

SUGGESTED_QUESTIONS = [
    "Summarize this document in a few points",
    "What are the main topics covered?",
    "List the key terms and their meanings",
]

QA_PROMPT_TEMPLATE = """You are a strict document question-answering assistant.
Answer the question using ONLY the context provided below.
If the context does not contain the information needed to answer the question,
reply with exactly this sentence and nothing else: I cannot find this in the document.
Answer the question and explicitly quote the exact sentence from the context that contains the answer.
Never use outside knowledge. Never guess. Never invent facts.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

CUSTOM_CSS = """
<style>
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.hero {
    background: linear-gradient(135deg, rgb(79, 70, 229) 0%, rgb(124, 58, 237) 55%, rgb(219, 39, 119) 100%);
    padding: 1.8rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 32px rgba(79, 70, 229, 0.28);
}
.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
    color: white;
    padding: 0;
}
.hero p {
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
    opacity: 0.92;
}
.doc-pill {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.35);
    font-size: 0.82rem;
    font-weight: 600;
}

.step-card {
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    height: 100%;
}
.step-card .step-icon {font-size: 2rem;}
.step-card .step-title {font-weight: 700; margin-top: 0.4rem;}
.step-card .step-text {font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem;}

.source-card {
    border-left: 4px solid rgb(124, 58, 237);
    background: rgba(124, 58, 237, 0.07);
    padding: 0.7rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.7rem;
    font-size: 0.85rem;
    line-height: 1.5;
}
.page-badge {
    display: inline-block;
    background: rgb(124, 58, 237);
    color: white;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
div[data-testid="stMetric"] {
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
}
section[data-testid="stSidebar"] .stButton > button,
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def extract_pages_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    page_texts = []
    page_metadatas = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        if extracted_text.strip():
            page_texts.append(extracted_text)
            page_metadatas.append({"page": page_number})
    return page_texts, page_metadatas, len(reader.pages)


def build_vector_store(page_texts, page_metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunk_documents = splitter.create_documents(page_texts, metadatas=page_metadatas)
    embeddings = load_embedding_model()
    vector_store = FAISS.from_documents(chunk_documents, embeddings)
    return vector_store, len(chunk_documents)


def build_conversational_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=GROQ_MODEL_NAME,
        temperature=0,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return chain


def initialize_session_state():
    defaults = {
        "chain": None,
        "messages": [],
        "document_name": None,
        "total_pages": 0,
        "total_chunks": 0,
        "pending_question": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_conversation():
    st.session_state.messages = []
    if st.session_state.chain is not None:
        st.session_state.chain.memory.clear()


def set_pending_question(question_text):
    st.session_state.pending_question = question_text


def build_chat_transcript():
    transcript_lines = [f"Chat about: {st.session_state.document_name}", ""]
    for message in st.session_state.messages:
        speaker = "You" if message["role"] == "user" else "Assistant"
        transcript_lines.append(f"{speaker}: {message['content']}")
        transcript_lines.append("")
    return "\n".join(transcript_lines)


def process_document(uploaded_pdf, groq_api_key):
    page_texts, page_metadatas, total_pages = extract_pages_from_pdf(uploaded_pdf)
    if not page_texts:
        st.error("No extractable text found. The PDF may be scanned or image-based.")
        return
    vector_store, total_chunks = build_vector_store(page_texts, page_metadatas)
    st.session_state.chain = build_conversational_chain(vector_store, groq_api_key)
    st.session_state.messages = []
    st.session_state.document_name = uploaded_pdf.name
    st.session_state.total_pages = total_pages
    st.session_state.total_chunks = total_chunks
    st.toast("Document is ready. Start asking questions!", icon="✅")


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free key from console.groq.com",
        )
        uploaded_pdf = st.file_uploader("📎 Upload your PDF", type=["pdf"])
        process_clicked = st.button(
            "🚀 Process Document", use_container_width=True, type="primary"
        )

        if process_clicked:
            if not groq_api_key:
                st.error("Please enter your Groq API key.")
            elif uploaded_pdf is None:
                st.error("Please upload a PDF file.")
            else:
                with st.spinner("Reading, chunking and indexing your document..."):
                    process_document(uploaded_pdf, groq_api_key)

        if st.session_state.document_name:
            st.divider()
            st.markdown("### 📊 Document Stats")
            left_column, right_column = st.columns(2)
            left_column.metric("Pages", st.session_state.total_pages)
            right_column.metric("Chunks", st.session_state.total_chunks)
            questions_asked = sum(
                1 for message in st.session_state.messages if message["role"] == "user"
            )
            st.metric("Questions asked", questions_asked)

            st.divider()
            if st.button("🧹 Clear Chat History", use_container_width=True):
                reset_conversation()
                st.rerun()
            if st.session_state.messages:
                st.download_button(
                    "💾 Download Chat",
                    data=build_chat_transcript(),
                    file_name="chat_history.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.divider()
        st.caption(f"Model: {GROQ_MODEL_NAME}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL_NAME.split('/')[-1]}")


def render_hero():
    document_pill = ""
    if st.session_state.document_name:
        safe_name = html.escape(st.session_state.document_name)
        document_pill = f'<div class="doc-pill">📄 {safe_name}</div>'
    st.markdown(
        f"""
        <div class="hero">
            <h1>📚 PagePilot </h1>
            <p>Ask questions about your PDF. Every answer is grounded in the document and backed by its sources.</p>
            {document_pill}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_screen():
    steps = [
        ("🔑", "1. Add API Key", "Paste your Groq API key in the sidebar."),
        ("📎", "2. Upload PDF", "Choose the document you want to chat with."),
        ("🚀", "3. Process", "Click Process Document and wait for indexing."),
    ]
    columns = st.columns(len(steps))
    for column, (icon, title, text) in zip(columns, steps):
        column.markdown(
            f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.info("👈 Complete the three steps in the sidebar to start chatting.")


def render_source_chunks(source_documents):
    with st.expander(f"📚 View sources ({len(source_documents)} chunks used)"):
        for chunk_index, document in enumerate(source_documents, start=1):
            page_number = document.metadata.get("page", "N/A")
            safe_text = html.escape(document.page_content).replace('\n', ' ')
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="page-badge">Chunk {chunk_index} · Page {page_number}</span><br>
                    {safe_text}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history():
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_source_chunks(message["sources"])


def render_suggested_questions():
    st.markdown("#### 💡 Try asking")
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for column, question_text in zip(columns, SUGGESTED_QUESTIONS):
        column.button(
            question_text,
            key=f"suggestion_{question_text}",
            on_click=set_pending_question,
            args=(question_text,),
            use_container_width=True,
        )


def handle_user_question(user_question):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🔎 Searching the document..."):
            try:
                response = st.session_state.chain.invoke({"question": user_question})
                answer_text = response["answer"]
                source_documents = response["source_documents"]
            except Exception as error:
                answer_text = f"⚠️ Something went wrong while generating the answer: {error}"
                source_documents = []
        st.markdown(answer_text)
        if source_documents:
            render_source_chunks(source_documents)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": source_documents}
    )


def main():
    st.set_page_config(page_title="PagePilot", page_icon="📚", layout="centered")
    initialize_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_sidebar()
    render_hero()

    if st.session_state.chain is None:
        render_welcome_screen()
        return

    render_chat_history()

    typed_question = st.chat_input("Ask anything about your document...")
    pending_question = st.session_state.pending_question
    st.session_state.pending_question = None
    user_question = typed_question or pending_question

    if not st.session_state.messages and not user_question:
        render_suggested_questions()

    if user_question:
        handle_user_question(user_question)
        st.rerun()


if __name__ == "__main__":
    main()