"""
Leibniz Agent - RAG Retrieval Testing Suite with Phi-2 Response Generation

Tests RAG system with Phi-2 LLM:
1. Knowledge base retrieval accuracy (FAISS)
2. Embedding model performance
3. Phi-2 response generation with retrieved context
4. Query-answer relevance scoring
5. End-to-end RAG pipeline testing
6. Performance metrics (retrieval + generation)

Pre-warms all models at startup (embedding model + Phi-2) for consistent performance.
"""

import time
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
import numpy as np
from datetime import datetime
import hashlib

# Setup comprehensive logging
log_filename = f"tara_rag_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Test queries for Leibniz University knowledge base
TEST_QUERIES = [
    "Tell me about the top courses in computer science and its marking scheme",
    "What are the admission requirements for the Master's program?",
    "How do I schedule an appointment with an academic advisor?",
    "What are the office hours for the admissions department?",
    "Tell me about the research opportunities at Leibniz University"
]

class LeibnizRAGTester:
    """Comprehensive RAG testing suite for Leibniz agent with Phi-2"""
    
    def __init__(self, knowledge_base_dir: str = "leibniz_knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.phi2_model_name = "microsoft/phi-2"
        self.embedder = None
        self.phi2_model = None
        self.phi2_tokenizer = None
        self.device = None
        self.faiss_index = None
        self.documents = []
        self.chunk_size = 800
        self.chunk_overlap = 150
        self.top_k = 5
        
        self.results = {
            "test_run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "embedding_model": self.embedding_model_name,
            "llm_model": self.phi2_model_name,
            "queries": [],
            "summary": {}
        }
        
        logging.info(f" Initializing Leibniz RAG Tester with Phi-2")
        logging.info(f" Knowledge base: {knowledge_base_dir}")
    
    def prewarm_models(self):
        """Pre-load and warm up all models at startup"""
        logging.info(f"\n{'='*60}")
        logging.info(f" PRE-WARMING MODELS")
        logging.info(f"{'='*60}")
        
        try:
            # 1. Load embedding model
            logging.info(f" Loading embedding model: {self.embedding_model_name}")
            start_time = time.time()
            self.embedder = SentenceTransformer(self.embedding_model_name)
            load_time = time.time() - start_time
            logging.info(f" Embedding model loaded in {load_time:.2f}s")
            
            # 2. Warm up embedding model with dummy query
            logging.info(f" Warming up embedding model...")
            start_time = time.time()
            _ = self.embedder.encode(["test query for warmup"], show_progress_bar=False)
            warmup_time = time.time() - start_time
            logging.info(f" Embedding model warmed up in {warmup_time:.2f}s")
        except Exception as e:
            logging.error(f" Embedding model loading failed: {e}")
            raise
        
        # Skip Phi-2 loading in fast retrieval mode
        logging.info(f" FAST MODE: Skipping Phi-2 model loading (retrieval only)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # 5. Load and index knowledge base
            logging.info(f" Loading knowledge base from {self.knowledge_base_dir}")
            self._load_knowledge_base()
            
            # 6. Build FAISS index
            logging.info(f" Building FAISS index...")
            start_time = time.time()
            self._build_faiss_index()
            index_time = time.time() - start_time
            logging.info(f" FAISS index built in {index_time:.2f}s")
            logging.info(f" Index size: {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logging.error(f" Knowledge base loading failed: {e}")
            raise
        
        logging.info(f"{'='*60}")
        logging.info(f" RAG SYSTEM PRE-WARMED AND READY (FAST MODE)")
        logging.info(f"{'='*60}\n")
    
    def _load_knowledge_base(self):
        """Load markdown files from Leibniz knowledge base directory"""
        if not os.path.exists(self.knowledge_base_dir):
            logging.error(f" Knowledge base directory not found: {self.knowledge_base_dir}")
            logging.error(f"Please ensure leibniz_knowledge_base exists!")
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_dir}")
        
        # Recursively find all .md files in subdirectories
        md_files = []
        for root, dirs, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        
        if not md_files:
            logging.error(f" No .md files found in {self.knowledge_base_dir}")
            raise FileNotFoundError(f"No markdown files in {self.knowledge_base_dir}")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Use relative path from knowledge base as source name
                    source_name = os.path.relpath(file_path, self.knowledge_base_dir)
                    chunks = self._chunk_text(content, source_name)
                    self.documents.extend(chunks)
            except Exception as e:
                logging.warning(f"️ Could not load {file_path}: {e}")
        
        logging.info(f" Loaded {len(self.documents)} chunks from {len(md_files)} files")
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        sentences = text.split('\n')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + "\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source
                    })
                current_chunk = sentence + "\n"
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source": source
            })
        
        return chunks
    
    def _build_faiss_index(self):
        """Build FAISS index from document chunks"""
        if not self.documents:
            logging.error(" No documents to index!")
            return
        
        # Extract text from documents
        texts = [doc["text"] for doc in self.documents]
        
        # Generate embeddings
        logging.info(f" Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
    
    def generate_response_with_phi2(self, query: str, retrieved_docs: List[Dict]) -> Tuple[str, float]:
        """Generate response using Phi-2 with retrieved context"""
        start_time = time.time()
        
        # Build context from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1} (from {doc['source']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs[:3])  # Use top 3 docs
        ])
        
        # Create prompt with Leibniz system context
        system_prompt = """You are a helpful AI assistant for Leibniz University Institute.
Answer questions based on the provided context documents.
Be concise, accurate, and professional."""
        
        full_prompt = f"""{system_prompt}

Context Documents:
{context_text}

Question: {query}

Answer:"""
        
        # Tokenize
        inputs = self.phi2_tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.phi2_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.phi2_tokenizer.pad_token_id,
                eos_token_id=self.phi2_tokenizer.eos_token_id
            )
        
        generation_time = (time.time() - start_time) * 1000  # ms
        
        # Decode response
        generated_text = self.phi2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text[len(full_prompt):].strip()
        
        return answer, generation_time
    
    def retrieve_documents(self, query: str) -> Tuple[List[Dict], float]:
        """Retrieve top-k relevant documents for query"""
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            self.top_k
        )
        
        # Get retrieved documents
        retrieved_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            retrieved_docs.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "distance": float(distance),
                "relevance_score": 1 / (1 + distance)  # Convert distance to similarity
            })
        
        retrieval_time = time.time() - start_time
        
        return retrieved_docs, retrieval_time * 1000  # Return ms
    
    def run_query_test(self, query: str, query_idx: int, total_queries: int):
        """Run complete RAG pipeline test for a single query"""
        logging.info(f"\n{'='*60}")
        logging.info(f"� Query {query_idx}/{total_queries}")
        logging.info(f"{'='*60}")
        logging.info(f" Question: {query}")
        
        try:
            # 1. Retrieve documents
            logging.info(f"\n Step 1: Retrieving relevant documents...")
            retrieved_docs, retrieval_time_ms = self.retrieve_documents(query)
            logging.info(f" Retrieved {len(retrieved_docs)} documents in {retrieval_time_ms:.2f}ms")
            
            # Show top 3 retrieved docs
            logging.info(f"\n Top 3 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                logging.info(f"   {i}. [{doc['source']}] Relevance: {doc['relevance_score']:.3f}")
                logging.info(f"      Preview: {doc['text'][:150]}...")
            
            # 2. Generate response with Phi-2
            logging.info(f"\n Step 2: Generating response with Phi-2...")
            answer, generation_time_ms = self.generate_response_with_phi2(query, retrieved_docs)
            logging.info(f" Generated response in {generation_time_ms:.2f}ms")
            
            # 3. Display answer
            logging.info(f"\n Generated Answer:")
            logging.info(f"   {answer}")
            
            # 4. Calculate total time
            total_time_ms = retrieval_time_ms + generation_time_ms
            logging.info(f"\n⏱️  Performance Metrics:")
            logging.info(f"   - Retrieval Time: {retrieval_time_ms:.2f}ms")
            logging.info(f"   - Generation Time: {generation_time_ms:.2f}ms")
            logging.info(f"   - Total Time: {total_time_ms:.2f}ms")
            
            # Store results
            test_result = {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "answer": answer,
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
                "total_time_ms": total_time_ms
            }
            
            self.results["queries"].append(test_result)
            
        except Exception as e:
            logging.error(f" Query test failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.results["queries"].append({
                "query": query,
                "error": str(e)
            })
    
    def run_all_tests(self):
        """Run complete RAG test suite with Phi-2"""
        logging.info(f"\n Starting Leibniz RAG Testing Suite with Phi-2")
        logging.info(f" Test Run ID: {self.results['test_run_id']}")
        
        start_time = time.time()
        
        # Pre-warm models (CRITICAL: Do this ONCE at startup)
        self.prewarm_models()
        
        # Run tests for each query
        for idx, query in enumerate(TEST_QUERIES, 1):
            self.run_query_test(query, idx, len(TEST_QUERIES))
        
        total_time = time.time() - start_time
        
        # Generate summary
        self._generate_summary(total_time)
        
        # Save results
        self._save_results()
        
        logging.info(f"\n Testing Complete! Total time: {total_time:.2f}s")
    
    def _generate_summary(self, total_time: float):
        """Generate test summary"""
        successful_tests = [q for q in self.results["queries"] if "error" not in q]
        failed_tests = [q for q in self.results["queries"] if "error" in q]
        
        if not successful_tests:
            logging.warning("️  No successful tests to summarize")
            return
        
        summary = {
            "total_time_s": total_time,
            "total_queries": len(TEST_QUERIES),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "avg_retrieval_time_ms": sum(q["retrieval_time_ms"] for q in successful_tests) / len(successful_tests),
            "avg_generation_time_ms": sum(q["generation_time_ms"] for q in successful_tests) / len(successful_tests),
            "avg_total_time_ms": sum(q["total_time_ms"] for q in successful_tests) / len(successful_tests)
        }
        
        self.results["summary"] = summary
        
        # Log summary
        logging.info(f"\n{'='*60}")
        logging.info(f" FINAL TEST SUMMARY (FAST RETRIEVAL MODE)")
        logging.info(f"{'='*60}")
        logging.info(f" Overall Performance:")
        logging.info(f"   - Total Queries: {summary['total_queries']}")
        logging.info(f"   - Successful: {summary['successful_tests']}")
        logging.info(f"   - Failed: {summary['failed_tests']}")
        logging.info(f"   - Avg Retrieval Time: {summary['avg_retrieval_time_ms']:.2f}ms")
        logging.info(f"   - Avg Total Time: {summary['avg_total_time_ms']:.2f}ms")
        logging.info(f"   - Total Test Time: {summary['total_time_s']:.2f}s")
        logging.info(f"{'='*60}\n")
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = f"leibniz_rag_phi2_results_{self.results['test_run_id']}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logging.info(f" Results saved to: {output_file}")


def main():
    """Main execution"""
    tester = LeibnizRAGTester(knowledge_base_dir="leibniz_knowledge_base")
    tester.run_all_tests()


if __name__ == "__main__":
    main()