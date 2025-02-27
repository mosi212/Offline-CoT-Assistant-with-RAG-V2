import os
import uuid
import hashlib
import torch
import threading
import logging
from typing import Tuple, Generator, List, Optional, Dict
from langchain.schema import SystemMessage
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

# LangChain and embedding imports for RAG functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader , PyPDFLoader
from sentence_transformers import SentenceTransformer
# New import for PDF loading
from langchain.memory import ConversationBufferMemory
import spacy

GLOBAL_NLP = spacy.load("en_core_web_sm")

# ============================================
# Global Configuration & Shared Components
# ============================================
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_NAME = "agentica-org/DeepScaleR-1.5B-Preview"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"
MODEL_PATH = os.path.join("./local_model", MODEL_NAME)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_PATH = "./local_embeddings"
CHROMA_PERSIST_DIR = "chroma_db"
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Global variables for heavy model components
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_EMBEDDINGS = None
GLOBAL_VECTORSTORE = None
GLOBAL_RETRIEVER = None


def init_global_components():
    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_EMBEDDINGS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER

    # Model loading with existence check
    model_files_exist = os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json"))
    if model_files_exist:
        try:
            logger.info(f"Loading local model from {MODEL_PATH}")
            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, device_map="auto", attn_implementation="sdpa", trust_remote_code=True
            )
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            logger.info("✅ Local model loaded")
        except Exception as e:
            logger.warning(f"Local model corrupted or incomplete: {e}. Downloading model.")
            model_files_exist = False

    if not model_files_exist:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", quantization_config=quant_config, torch_dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=True
        )
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        os.makedirs(MODEL_PATH, exist_ok=True)
        GLOBAL_MODEL.save_pretrained(MODEL_PATH)
        GLOBAL_TOKENIZER.save_pretrained(MODEL_PATH)
        logger.info("💾 Model downloaded and cached")

    # Embeddings loading with specific file check
    embedding_file = os.path.join(EMBEDDING_MODEL_PATH, "pytorch_model.bin")
    os.makedirs(EMBEDDING_MODEL_PATH, exist_ok=True)
    if not os.path.exists(embedding_file):
        logger.info("⬇️ Downloading embedding model...")
        st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st_model.save(EMBEDDING_MODEL_PATH)
        logger.info("💾 Saved embeddings locally")
    else:
        logger.info("✅ Loading embeddings from local cache")
        st_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={"device": "cpu"})

    # Vectorstore and retriever initialization (unchanged)
    GLOBAL_VECTORSTORE = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=GLOBAL_EMBEDDINGS)
    doc_count = GLOBAL_VECTORSTORE._collection.count()
    k = min(4, doc_count) if doc_count > 0 else 1
    GLOBAL_RETRIEVER = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": k})
    logger.info("Global components initialized.")


# ============================================
# Sentence-based splitting function
# ============================================
def sentence_based_split(text: str, target_word_count: int = 256, overlap_words: int = 32, max_tokens: int = 512) -> \
List[str]:
    doc = GLOBAL_NLP(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk_words = []
    tokenizer = GLOBAL_TOKENIZER  # Use global tokenizer for token counting

    for sentence in sentences:
        words = sentence.split()
        # Split long sentences
        if len(words) > 100:
            sub_sentences = [sentence[i:i + 100] for i in range(0, len(words), 100)]
            words = " ".join(sub_sentences).split()

        if len(current_chunk_words) + len(words) > target_word_count:
            chunk_text = " ".join(current_chunk_words)
            if len(tokenizer.encode(chunk_text)) <= max_tokens:
                chunks.append(chunk_text)
            else:
                # Truncate if exceeds max_tokens
                truncated_words = current_chunk_words
                while len(tokenizer.encode(" ".join(truncated_words))) > max_tokens:
                    truncated_words.pop()
                chunks.append(" ".join(truncated_words))
            if len(current_chunk_words) >= overlap_words:
                current_chunk_words = current_chunk_words[-overlap_words:]
            current_chunk_words.extend(words)
        else:
            current_chunk_words.extend(words)

    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        if len(tokenizer.encode(chunk_text)) <= max_tokens:
            chunks.append(chunk_text)
        else:
            truncated_words = current_chunk_words
            while len(tokenizer.encode(" ".join(truncated_words))) > max_tokens:
                truncated_words.pop()
            chunks.append(" ".join(truncated_words))
    return chunks


# ============================================
# CoT Agent with Integrated RAG, System Message and Chat Memory
# ============================================
class CoTAgent:
    def __init__(
            self,
            memory: Optional[ConversationBufferMemory] = None,
            model=None,
            tokenizer=None,
            embeddings=None,
            vectorstore=None,
            retriever=None
    ):
        self.logger = logger
        # Use shared global components if provided
        self.model = model if model is not None else GLOBAL_MODEL
        self.tokenizer = tokenizer if tokenizer is not None else GLOBAL_TOKENIZER
        self.embeddings = embeddings if embeddings is not None else GLOBAL_EMBEDDINGS
        self.vectorstore = vectorstore if vectorstore is not None else GLOBAL_VECTORSTORE
        self.retriever = retriever if retriever is not None else GLOBAL_RETRIEVER
        self.retriever_cache = {}
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Enhanced system message
        self.system_message = (
            "You are an AI assistant designed to solve problems and answer questions with clear, step-by-step reasoning. "
            "Your goal is to provide accurate, comprehensive, and well-justified responses, leveraging retrieved context when applicable.\n\n"
            "When responding to a task or question, follow these guidelines:\n"
            "1. **Clarify the Question:** Ensure you understand the query. Restate it in your own words if it helps confirm your interpretation.\n"
            "2. **Assess Retrieved Context:** Evaluate the provided context from retrieved documents. Use it only if it is directly relevant and reliable; otherwise, rely on your general knowledge.\n"
            "3. **Break Down the Problem:** Divide complex tasks into smaller, manageable parts or identify key objectives.\n"
            "4. **Explore Options:** Consider multiple approaches or answers, weighing their strengths and weaknesses.\n"
            "5. **Build the Solution:** Combine insights from context (if relevant) and your analysis to develop a clear, logical response. Refine it if needed.\n"
            "6. **Verify the Response:** Double-check your reasoning and ensure the answer is consistent and sensible.\n"
            "7. **Present Clearly:** Provide your final answer with a concise summary of your reasoning steps for justification.\n\n"
            "At each step, briefly explain your thought process to make your reasoning transparent and easy to follow."
        )
        # Revised chain-of-thought template
        self.cot_template = (
            "Task/Question: {question}\n\n"
            "Relevant Context: {content}\n\n"
        )
        # Initialize memory
        self.memory = memory if memory is not None else ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Insert the system message only once at conversation start if not already present.
        if not self.memory.buffer_as_str:
            # Insert as a system message at the beginning of the conversation history.
            self.memory.chat_memory.messages.insert(0, SystemMessage(content=self.system_message))
        self.current_urls = []
        self.update_urls(DEFAULT_URLS)

    def _process_documents(self, docs):
        from langchain.docstore.document import Document  # Import Document class
        new_docs = []
        for doc in docs:
            chunks = sentence_based_split(doc.page_content, target_word_count=128, overlap_words=32)
            for chunk in chunks:
                new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
                clean_content = new_doc.page_content.strip()
                new_doc.metadata.update({
                    "content_hash": hashlib.sha256(clean_content.encode()).hexdigest(),
                    "source": new_doc.metadata.get("source", "unknown")
                })
                new_docs.append(new_doc)
        existing_hashes = set()
        if self.vectorstore._collection.count() > 0:
            existing_data = self.vectorstore._collection.get(include=["metadatas"])
            existing_hashes = {m["content_hash"] for m in existing_data["metadatas"]}
        final_docs = [doc for doc in new_docs if doc.metadata["content_hash"] not in existing_hashes]
        if final_docs:
            self.logger.info(f"🆕 Adding {len(final_docs)} new documents")
            self.vectorstore.add_documents(final_docs)
        doc_count = self.vectorstore._collection.count()
        k = min(4, doc_count) if doc_count > 0 else 1
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

    def update_urls(self, new_urls: List[str]):
        if set(new_urls) != set(self.current_urls):
            self.logger.info("🔄 Updating document sources")
            self.current_urls = new_urls
            loader = WebBaseLoader(
                web_paths=new_urls,
                requests_kwargs={"headers": {"User-Agent": f"rag-agent/{uuid.uuid4()}"}}
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = doc.metadata.get("source", "url")
            self._process_documents(docs)
        else:
            self.logger.info("✅ URLs unchanged")

    def ingest_pdf_documents(self, file_paths: List[str]):
        all_docs = []
        for path in file_paths:
            self.logger.info(f"Ingesting PDF: {path}")
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                all_docs.extend(docs)
            except Exception as e:
                self.logger.error(f"Error processing PDF {path}: {e}")
        if all_docs:
            self._process_documents(all_docs)
            return f"Uploaded and processed {len(all_docs)} documents from PDFs."
        else:
            return "No valid PDF documents were processed."

    def _stream_generate_response(self, prompt: str, max_tokens: int) -> Generator[str, None, None]:
        self.logger.info("Starting streaming generation.")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id  # Sync model config
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
            "num_beams": 1,
            "streamer": streamer,
        }
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            formatted_response = self._format_response(full_response)
            yield formatted_response
        thread.join()
        self.logger.info("Streaming generation complete.")
        # Update memory with the conversation pair
        self.memory.chat_memory.add_user_message(self.last_question)
        self.memory.chat_memory.add_ai_message(full_response)
        yield full_response  # Final yield with complete answer

    def _format_response(self, text: str) -> str:
        lines = text.splitlines()
        formatted_lines = []
        in_list = False
        for line in lines:
            line = line.strip()
            if line.startswith(tuple(f"{i}." for i in range(1, 10))):
                formatted_lines.append(f"**Step {line.split('.')[0]}** - {line.split('.', 1)[1].strip()}")
                in_list = True
            elif in_list and line:
                formatted_lines.append(f"- {line}")
            else:
                formatted_lines.append(line)
                in_list = False
        return "\n".join(formatted_lines)

    def generate_with_cot(self, question: str, urls: Optional[List[str]] = None) -> Generator[str, None, None]:
        self.last_question = question
        if urls:
            self.update_urls(urls)
        cache_key = hashlib.sha256(question.encode()).hexdigest()
        if cache_key in self.retriever_cache:
            docs = self.retriever_cache[cache_key]
            self.logger.info("Using cached retriever results.")
        else:
            try:
                docs = self.retriever.invoke(question)
                self.retriever_cache[cache_key] = docs
            except Exception as e:
                self.logger.error(f"Error retrieving documents: {e}")
                docs = []
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""
        context_str = f"\n\nRelevant Context (use only if directly applicable to the question):\n{context}" if context else ""

        # Get recent history, excluding the system message
        recent_history = self.memory.chat_memory.messages[1:]  # Skip system message
        history_str = "\n".join([msg.content for msg in recent_history[-5:]]) if recent_history else ""

        # Construct prompt with system message always included
        combined_prompt = self.system_message + "\n\n" + self.cot_template.format(question=question,
                                                                                  content=context_str)
        if history_str:
            combined_prompt += "\n\nRecent Chat History:\n" + history_str

        # Truncate if exceeding token limit
        max_input_tokens = 7000
        tokenized_prompt = self.tokenizer.encode(combined_prompt)
        if len(tokenized_prompt) > max_input_tokens:
            # Truncate context first
            excess = len(tokenized_prompt) - max_input_tokens
            context_lines = context_str.splitlines()
            while excess > 0 and context_lines:
                context_lines.pop()
                context_str = "\n".join(context_lines)
                combined_prompt = self.system_message + "\n\n" + self.cot_template.format(question=question,
                                                                                          content=context_str)
                if history_str:
                    combined_prompt += "\n\nRecent Chat History:\n" + history_str
                tokenized_prompt = self.tokenizer.encode(combined_prompt)
                excess = len(tokenized_prompt) - max_input_tokens

        self.logger.info(f"Final prompt:\n{combined_prompt}")
        yield from self._stream_generate_response(combined_prompt, max_tokens=8192)


# ============================================
# Global Agent Factory for Session Isolation
# ============================================
def get_agent(existing_agent: Optional[CoTAgent] = None) -> CoTAgent:
    if existing_agent is None:
        # Ensure global components are initialized
        global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_EMBEDDINGS, GLOBAL_VECTORSTORE, GLOBAL_RETRIEVER
        if GLOBAL_MODEL is None:
            init_global_components()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return CoTAgent(
            memory=memory,
            model=GLOBAL_MODEL,
            tokenizer=GLOBAL_TOKENIZER,
            embeddings=GLOBAL_EMBEDDINGS,
            vectorstore=GLOBAL_VECTORSTORE,
            retriever=GLOBAL_RETRIEVER
        )
    return existing_agent


# ============================================
# Gradio Interface Functions with Session State
# ============================================
def chat_interface(message: str, urls_text: str, agent_state: dict) -> Generator[str, None, None]:
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()] if urls_text.strip() else None
    agent = agent_state.get("agent")
    if agent is None:
        agent = get_agent()
        agent_state["agent"] = agent
    try:
        yield from agent.generate_with_cot(message, urls)
    except Exception as e:
        logger.exception("Generation error:")
        error_msg = (
            f"❌ An error occurred while processing your request.\n"
            f"**Details:** {str(e)}\n"
            f"Please try again or simplify your question. If the issue persists, check the URLs or uploaded files."
        )
        yield error_msg


def ingest_urls(urls_text: str) -> str:
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
    if not urls:
        return "No valid URLs provided."
    global_agent = get_agent()
    global_agent.update_urls(urls)
    return f"Updated document sources with {len(urls)} URL(s)."


def ingest_pdfs(files: List) -> str:
    file_paths = []
    os.makedirs("temp_uploads", exist_ok=True)
    for file in files:
        if isinstance(file, str):
            file_paths.append(file)
        else:
            temp_path = os.path.join("temp_uploads", file.name)
            with open(temp_path, "wb") as f:
                f.write(file.read())
            file_paths.append(temp_path)
    global_agent = get_agent()
    result = global_agent.ingest_pdf_documents(file_paths)
    return result


# ============================================
# Gradio UI Setup with Session State for Independent Clients
# ============================================
with gr.Blocks(title="Offline CoT Assistant with RAG and Per-Session Memory") as interface:
    gr.Markdown("# Offline CoT Assistant with RAG and Per-Session Memory")
    gr.Markdown(
        "A chain-of-thought agent augmented with retrieval (RAG) and individual chat memory for each user/browser tab.")

    agent_state = gr.State(value={})

    with gr.Tabs():
        with gr.Tab("Chat Interface"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_input = gr.Textbox(lines=4, placeholder="Enter your question...", label="Your Question")
                    urls_optional = gr.Textbox(label="Optional: Enter URLs (one per line)", lines=3)
                    with gr.Row():
                        submit_btn = gr.Button("Get Answer")
                        reset_btn = gr.Button("Reset Chat")
                with gr.Column(scale=2):
                    answer_output = gr.Markdown(label="Answer")
                    feedback = gr.Slider(1, 5, step=1, label="Rate this response (1-5)")
            submit_btn.click(fn=chat_interface, inputs=[user_input, urls_optional, agent_state], outputs=answer_output)
            reset_btn.click(
                fn=lambda state: ({"agent": get_agent()}, ""),
                inputs=agent_state,
                outputs=[agent_state, answer_output]
            )

        with gr.Tab("URL Ingestion"):
            gr.Markdown("### Update Document Sources via URLs")
            url_input = gr.Textbox(
                label="Enter URLs (one per line)",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                lines=5
            )
            ingest_button = gr.Button("Update Document Sources")
            ingest_status = gr.Textbox(label="Status")
            ingest_button.click(fn=ingest_urls, inputs=url_input, outputs=ingest_status)

        with gr.Tab("PDF Upload"):
            gr.Markdown("### Upload PDF Documents")
            pdf_files = gr.File(label="Select PDF files", file_count="multiple", file_types=[".pdf"])
            pdf_ingest_button = gr.Button("Ingest PDFs")
            pdf_status = gr.Textbox(label="PDF Ingestion Status")
            pdf_ingest_button.click(fn=ingest_pdfs, inputs=pdf_files, outputs=pdf_status)

    interface.css = """
    .input-box textarea { font-size: 16px !important; padding: 15px !important; }
    .process-box { background: #1b1c1b; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; }
    footer { display: none !important; }
    """

if __name__ == "__main__":
    try:
        interface.launch(server_name="0.0.0.0", server_port=8000, show_error=True, share=False)
    except OSError as e:
        logger.error(f"Port error: {e}")
        interface.launch(server_name="0.0.0.0", server_port=0, show_error=True)
