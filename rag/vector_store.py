import os
import threading
import json
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

from langchain_core.embeddings import Embeddings
from typing import List

class CachedEmbeddings(Embeddings):
    """
    Wrapper for embeddings to cache results locally.
    Avoids re-computing embeddings for identical text.
    """
    def __init__(self, wrapped_embeddings, cache_path="rag/embeddings_cache.json"):
        self.wrapped = wrapped_embeddings
        self.cache_path = cache_path
        self.cache = {}
        self._load_cache()
        self._lock = threading.Lock()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except: self.cache = {}
            
    def _save_cache(self):
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
            except Exception as e:
                print(f"Failed to save embedding cache: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache
        for i, text in enumerate(texts):
            h = hashlib.md5(text.encode()).hexdigest()
            if h in self.cache:
                results.append(self.cache[h])
            else:
                results.append(None) # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Compute missing
        if texts_to_embed:
            print(f"Computing embeddings for {len(texts_to_embed)} new items...")
            new_embeddings = self.wrapped.embed_documents(texts_to_embed)
            
            for idx, emb, text in zip(indices_to_embed, new_embeddings, texts_to_embed):
                results[idx] = emb
                h = hashlib.md5(text.encode()).hexdigest()
                self.cache[h] = emb
            
            # Save incrementally
            self._save_cache()
            
        return results

    def embed_query(self, text: str) -> List[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        
        emb = self.wrapped.embed_query(text)
        self.cache[h] = emb
        self._save_cache()
        return emb

class VectorStoreManager:
    _embeddings = None
    _lock = threading.Lock()

    def __init__(self, index_path: str = "rag/faiss_index"):
        self.index_path = index_path
        if VectorStoreManager._embeddings is None:
            # Load embeddings model once
            base_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # Wrap with caching
            VectorStoreManager._embeddings = CachedEmbeddings(base_embeddings)
            
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

