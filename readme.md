# Offline CoT Assistant with RAG & Per-Session Memory

**Offline CoT Assistant** is a state-of-the-art AI assistant that leverages chain-of-thought (CoT) reasoning, retrieval-augmented generation (RAG), and per-session memory to provide clear, step-by-step answers. This project integrates multiple cutting-edge technologies—from transformers and LangChain to Gradio for a sleek, interactive UI—making it ideal for offline deployment and enterprise-grade solutions.

## Key Features

- **Chain-of-Thought Reasoning:** Provides clear, step-by-step explanations for complex queries.
- **Retrieval-Augmented Generation (RAG):** Seamlessly integrates with web-based and PDF document ingestion to enrich responses with external context.
- **Per-Session Memory:** Each session maintains its own conversation history, ensuring contextual and personalized interactions.
- **Multi-Modal Ingestion:**
  - **URL Ingestion:** Fetch and process content from multiple URLs.
  - **PDF Upload:** Directly ingest and extract information from PDF files.
- **Local Caching:** Caches large language models and embeddings for faster, offline performance.
- **User-Friendly Gradio Interface:** An interactive web UI that supports live streaming of generated responses.

## Demo

After installing the project, simply run the application and open your browser to interact with the assistant:

```bash
python main.py
```

Then navigate to: [http://0.0.0.0:8000](http://0.0.0.0:8000)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mosi212/Offline-CoT-Assistant-with-RAG-V2.git
   cd <your-repo-name>
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download necessary NLP models:**

   For Spacy, run:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

- **Chat Interface:** Ask questions and receive detailed, step-by-step responses.
- **URL Ingestion:** Update document sources by providing one URL per line.
- **PDF Upload:** Upload PDF documents to augment the assistant’s knowledge base.

Simply launch the app:

```bash
python main.py
```

And interact with the UI via your browser.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, enhancements, or additional features.

## License

This project is licensed under the [MIT License](LICENSE).
