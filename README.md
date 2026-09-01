# 📄 Smart PDF RAG Assistant

A Document Retrieval-Augmented Generation (RAG) web application that enables interactive, grounded natural language question-answering over custom PDF files.

🚀 **Live Demo:** [smart-pdf-rag.streamlit.app](https://smart-pdf-rag-hm7kgreilkduzajdrpv3dx.streamlit.app)

---

## 🌟 Key Features
* **Semantic Vector Search:** Leverages `sentence-transformers` (`all-MiniLM-L6-v2`) and ChromaDB to perform fast similarity searches over chunked PDF text.
* **Context-Grounded QA:** Synthesizes precise, context-aware answers powered by Google's `gemini-3.6-flash` model via the official `google-genai` SDK.
* **Strict Guardrails:** Custom prompt engineering prevents hallucinations by returning a strict fallback message when questions fall outside the uploaded document's context.
* **Interactive UI:** Built with Streamlit for seamless PDF uploads, document processing state management, and continuous chat history.

---

## 🛠️ Tech Stack
* **Language:** Python
* **LLM:** Google Gemini (`gemini-3.6-flash`)
* **Framework:** LangChain (LCEL)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace / `sentence-transformers`
* **UI & Deployment:** Streamlit Community Cloud

---


