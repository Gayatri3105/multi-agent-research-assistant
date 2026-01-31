# ChromaDB + embeddings

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Union
import uuid
import hashlib

class Memory:
    def __init__(self):
        # Use PersistentClient to save data across sessions
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("research_assistant")
        self._model = None  # Lazy load the model

    @property
    def model(self):
        """Lazy load the embedding model only when needed"""
        if self._model is None:
            try:
                print("Loading SentenceTransformer model...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                print("Memory features will be disabled.")
                # Return a dummy model that raises an error if used
                raise RuntimeError("Embedding model not available. Check your internet connection.")
        return self._model

    def save(self, data: Union[str, List[str]], query: str = None, metadata: dict = None):
        """
        Save text or list of texts to memory.
        
        Args:
            data: Single text string or list of text strings
            query: Optional query that generated this data
            metadata: Optional metadata dict
        """
        try:
            # Convert single string to list for uniform processing
            texts = [data] if isinstance(data, str) else data
            
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for text in texts:
                if not text or not text.strip():
                    continue
                    
                embedding = self.model.encode(text).tolist()
                embeddings.append(embedding)
                documents.append(text)
                
                # Create metadata
                meta = metadata.copy() if metadata else {}
                if query:
                    meta["query"] = query
                metadatas.append(meta)
                
                # Generate unique ID from text hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                ids.append(f"doc_{text_hash}")
            
            if embeddings:
                try:
                    self.collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    # Handle duplicate IDs gracefully
                    print(f"Memory save warning: {e}")
        except RuntimeError as e:
            print(f"Cannot save to memory: {e}")
            return

    def search(self, query: str, k: int = 3):
        """
        Search for similar documents in memory.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document strings, or empty list if no results
        """
        try:
            # Check if collection is empty
            count = self.collection.count()
            if count == 0:
                return []
            
            embedding = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count)
            )
            
            # Extract documents from results
            if results and "documents" in results and results["documents"]:
                return results["documents"][0]  # First query's results
            return []
        except RuntimeError as e:
            print(f"Cannot search memory: {e}")
            return []
        except Exception as e:
            print(f"Memory search error: {e}")
            return []
    
    def clear(self):
        """Clear all data from memory."""
        try:
            self.client.delete_collection("research_assistant")
            self.collection = self.client.get_or_create_collection("research_assistant")
        except Exception as e:
            print(f"Memory clear error: {e}")