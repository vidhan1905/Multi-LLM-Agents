import os
import nltk
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from core.llm_manager import LLMManager

class RAGAgent:
    """Agent that can answer questions based on a collection of documents."""

    def __init__(self, llm_manager: LLMManager, docs_dir: str = "documents"):
        """
        Initializes the RAGAgent.
        
        Args:
            llm_manager: An instance of LLMManager to get the configured LLM.
            docs_dir: The directory containing the documents to load.
        """
        # self._download_nltk_data()
        self.llm = llm_manager.get_langchain_llm()
        self.embeddings = llm_manager.get_langchain_embeddings()
        
        print("Loading and processing documents...")
        self.retriever = self._create_retriever(docs_dir)
        
        if self.retriever:
            self.retrieval_chain = self._create_chain()
            print("RAG Agent is ready.")
        else:
            self.retrieval_chain = None
            print("Warning: RAG Agent is not available because no documents were found or processed.")

    def _download_nltk_data(self):
        """Downloads the necessary NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK 'punkt' model...")
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading NLTK 'averaged_perceptron_tagger' model...")
            nltk.download('averaged_perceptron_tagger')

    def _create_retriever(self, docs_dir: str):
        """Loads PDF documents, splits them, and creates a FAISS retriever."""
        try:
            if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
                print(f"Directory '{docs_dir}' is empty or does not exist.")
                return None

            pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
            if not pdf_files:
                print("No PDF files found in the documents directory.")
                return None

            all_docs = []
            for pdf_file in pdf_files:
                file_path = os.path.join(docs_dir, pdf_file)
                print(f"Loading document: {file_path}")
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())

            if not all_docs:
                print("No content could be loaded from the PDF files.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            print(f"Error creating retriever: {e}")
            return None

    def _create_chain(self):
        """Creates the retrieval chain for answering questions."""
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, just say that you "
            "don't know. Provide a detailed and comprehensive answer based on the context.\\n\\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def run(self, query: str):
        """
        Runs a query against the document collection.
        
        Args:
            query: The question to ask.
            
        Returns:
            The agent's answer.
        """
        if self.retrieval_chain is None:
            return "RAG Agent is not available. Please add documents to the 'documents' directory."
            
        try:
            result = self.retrieval_chain.invoke({"input": query})
            return result.get("answer", "No answer could be generated.")
        except Exception as e:
            return f"An error occurred: {e}"
