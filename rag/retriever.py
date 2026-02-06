from typing import List, Dict, Optional
from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager

class RAGRetriever:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_store = self.vector_manager.load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def reload(self):
        """Re-read the vector store from disk."""
        self.vector_store = self.vector_manager.load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def search(self, query: str, filters: Optional[Dict] = None, k: int = 5) -> List[Document]:
        """
        Search for documents with robust metadata filtering.
        """
        search_kwargs = {"k": k}
        
        filter_fn = None
        if filters:
            def filter_fn(metadata):
                for key, value in filters.items():
                    met_val = metadata.get(key)
                    if met_val is None:
                        return False
                    
                    met_val_str = str(met_val).lower().strip()
                    
                    if isinstance(value, list):
                        norm_values = [str(v).lower().strip() for v in value]
                        match_found = False
                        for v_item in norm_values:
                            if key in ["insurer", "insurance_type"]:
                                # Flexible match for categories (containment)
                                if v_item in met_val_str or met_val_str in v_item:
                                    match_found = True
                                    break
                            else:
                                # Fuzzy/substring match for other fields (like product_name)
                                if v_item in met_val_str or met_val_str in v_item:
                                    match_found = True
                                    break
                        if not match_found:
                            return False
                    else:
                        norm_value = str(value).lower().strip()
                        if key in ["insurer", "insurance_type"]:
                            if norm_value not in met_val_str and met_val_str not in norm_value:
                                return False
                        else:
                            if norm_value not in met_val_str and met_val_str not in norm_value:
                                return False
                return True
            
            search_kwargs["filter"] = filter_fn
            
        
        if filter_fn:
            # Compensate for post-retrieval filtering by increasing search depth
            # Increased to handle indices with many chunks per plan
            k_expanded = max(k * 500, 3000)
            search_kwargs["k"] = k_expanded
            search_kwargs["fetch_k"] = k_expanded * 2
            
        results = self.vector_store.similarity_search(query, **search_kwargs)
        return results[:k]

    def comparative_search(self, query: str, products: List[str]) -> Dict[str, List[Document]]:
        """
        Retrieves documents for the same query across multiple products for comparison.
        """
        results = {}
        for product in products:
            product_results = self.search(query, filters={"product_name": product}, k=3)
            results[product] = product_results
            
        return results

