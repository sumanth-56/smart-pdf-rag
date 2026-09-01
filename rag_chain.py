import os
from google import genai
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """Combines retrieved document content into a single deduplicated string."""
    unique_contents = list(dict.fromkeys(doc.page_content for doc in docs))
    return "\n\n".join(unique_contents)

class GeminiLLM:
    """Direct wrapper for Google GenAI SDK using gemini-3.6-flash."""
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def __call__(self, prompt_text: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt_text,
        )
        return response.text

def get_rag_chain(vector_store):
    """Constructs a RAG chain connecting ChromaDB vector store with Gemini."""
    llm_func = GeminiLLM()

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    template = """Answer the question based strictly on the context below. If the answer is not in the context, output "I cannot find the answer in the provided document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | (lambda p: llm_func(p.to_string()))
    )

    return rag_chain, retriever