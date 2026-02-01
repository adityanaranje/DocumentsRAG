import os
import threading
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

class VectorStoreManager:
    _embeddings = None
    _lock = threading.Lock()

    def __init__(self, index_path: str = "rag/faiss_index"):
        self.index_path = index_path
        if VectorStoreManager._embeddings is None:
            # Load embeddings model once
            VectorStoreManager._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        self.embeddings = VectorStoreManager._embeddings
    
    def create_vector_store(self, documents: List[Document], batch_size: int = 100):
        """
        Creates a new FAISS index from the provided documents and saves it locally.
        Uses batching and tqdm to show progress.
        """
        if not documents:
            print("No documents to index.")
            return

        from tqdm import tqdm
        print(f"Creating vector store with {len(documents)} chunks...")
        
        # Initialize with first batch
        first_batch = documents[:batch_size]
        vector_store = FAISS.from_documents(first_batch, self.embeddings)
        
        # Add remaining batches with progress bar
        if len(documents) > batch_size:
            for i in tqdm(range(batch_size, len(documents), batch_size), desc="Creating Vectors"):
                batch = documents[i : i + batch_size]
                vector_store.add_documents(batch)
        
        # Save to disk
        with VectorStoreManager._lock:
            vector_store.save_local(self.index_path)
        print(f"Vector store saved to {self.index_path}")
        return vector_store

    def load_vector_store(self):
        """
        Loads the existing FAISS index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run ingestion first.")
        
        with VectorStoreManager._lock:
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True 
            )

    def update_vector_store(self, documents: List[Document], batch_size: int = 100):
        """
        Loads an existing index, adds new documents, and saves.
        Uses batching and tqdm to show progress.
        """
        if not documents:
            return

        if not os.path.exists(self.index_path):
            print("Index doesn't exist. Creating new one...")
            return self.create_vector_store(documents, batch_size=batch_size)

        from tqdm import tqdm
        print(f"Updating existing index with {len(documents)} new chunks...")
        vector_store = self.load_vector_store()
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Updating Vectors"):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(batch)
            
        with VectorStoreManager._lock:
            vector_store.save_local(self.index_path)
        print(f"Update complete. Saved to {self.index_path}")

    def delete_documents_by_source(self, source_path: str):
        """
        Removes all documents from the index that match the given source path.
        """
        if not os.path.exists(self.index_path):
            return

        vector_store = self.load_vector_store()
        
        # Identify IDs to delete
        ids_to_delete = []
        # docstore._dict is {id: Document}
        for doc_id, doc in vector_store.docstore._dict.items():
            # Check source match (handle both absolute and relative discrepancies if needed)
            # source_path should be consistent with how it was ingested
            if doc.metadata.get("source") == source_path:
                ids_to_delete.append(doc_id)
        
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} chunks for source: {source_path}")
            with VectorStoreManager._lock:
                vector_store.delete(ids_to_delete)
                vector_store.save_local(self.index_path)
            print("Deletion complete and index saved.")
        else:
            print(f"No documents found for source: {source_path}")

