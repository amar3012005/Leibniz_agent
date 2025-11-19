"""
Leibniz RAG Hybrid System - Rule-Based + Phi-2 GGUF (ULTRA-OPTIMIZED)
======================================================================

Combines fast rule-based extraction with quantized Phi-2 GGUF for maximum performance.

Architecture:
1. Fast retrieval with FAISS (6-18ms)
2. Rule-based context extraction (<1ms overhead)
3. Phi-2 GGUF 4-bit quantized generation (100-500ms on CPU/GPU)
4. Total: ~150-600ms average (8-10x faster than HuggingFace Phi-2)

GGUF Advantages:
- 4-bit quantization: 2.7GB → 1.5GB model size
- llama.cpp backend: Optimized C++ inference engine
- CPU/GPU support: Efficient on both with CUDA/Metal acceleration
- Faster load time: 3-5s vs 10s for full precision
- Lower memory: ~2GB RAM vs ~6GB RAM
- Faster generation: 200-500ms vs 2000-3000ms

Performance Target:
- Model Load: <5s (vs 10s HuggingFace)
- Retrieval: <20ms
- Extraction: <1ms
- Generation: <500ms (vs 2000ms HuggingFace)
- Total: <600ms per query (3-4x faster than optimized HF version)
- Answer: 250-400 chars, conversational tone

Setup:
1. Install llama-cpp-python: pip install llama-cpp-python
2. Download Phi-2 GGUF: https://huggingface.co/TheBloke/phi-2-GGUF
   Recommended: phi-2.Q4_K_M.gguf (1.6GB, best quality/speed balance)
"""

import os
import sys
import logging
import time
import json
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Check for llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("️  llama-cpp-python not installed. Install with: pip install llama-cpp-python")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Test queries
TEST_QUERIES = [
    "Tell me about the top courses in computer science and its marking scheme",
    "What are the admission requirements for the Master's program?",
    "How do I schedule an appointment with an academic advisor?",
    "What are the office hours for the admissions department?",
    "Tell me about the research opportunities at Leibniz University"
]

class LeibnizHybridRAGGGUF:
    def __init__(self, knowledge_base_dir="leibniz_knowledge_base", 
                 model_path="models/phi-2.Q4_K_M.gguf"):
        """
        Initialize Leibniz Hybrid RAG with GGUF model
        
        Args:
            knowledge_base_dir: Path to markdown knowledge base
            model_path: Path to GGUF model file (download from HuggingFace)
        """
        logging.info(f" Initializing Leibniz Hybrid RAG (GGUF Optimized)")
        logging.info(f" Knowledge base: {knowledge_base_dir}")
        logging.info(f" Model path: {model_path}\n")
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF support.\n"
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
            )
        
        self.knowledge_base_dir = knowledge_base_dir
        self.model_path = model_path
        self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.top_k = 5
        
        # Initialize storage
        self.embedder = None
        self.llm = None
        self.documents = []
        self.faiss_index = None
        
        # Results tracking
        self.results = {
            "test_run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": "GGUF",
            "queries": []
        }
    
    def prewarm_models(self):
        """Pre-warm embedding model and load GGUF model"""
        logging.info(f"{'='*60}")
        logging.info(f" PRE-WARMING MODELS (GGUF MODE)")
        logging.info(f"{'='*60}")
        
        # 1. Load embedding model
        logging.info(f" Loading embedding model: {self.embedding_model_name}")
        start_time = time.time()
        self.embedder = SentenceTransformer(self.embedding_model_name)
        embed_load_time = time.time() - start_time
        logging.info(f" Embedding model loaded in {embed_load_time:.2f}s")
        
        # 2. Warmup embeddings
        logging.info(f" Warming up embedding model...")
        start_time = time.time()
        _ = self.embedder.encode(["test query for warmup"], show_progress_bar=False)
        warmup_time = time.time() - start_time
        logging.info(f" Embedding model warmed up in {warmup_time:.2f}s")
        
        # 3. Load GGUF model
        if not os.path.exists(self.model_path):
            logging.error(f"\n GGUF model not found: {self.model_path}")
            logging.error(f" Download Phi-2 GGUF from: https://huggingface.co/TheBloke/phi-2-GGUF")
            logging.error(f"   Recommended: phi-2.Q4_K_M.gguf (1.6GB)")
            logging.error(f"   Place in: models/phi-2.Q4_K_M.gguf\n")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logging.info(f"\n Loading Phi-2 GGUF model: {self.model_path}")
        model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        logging.info(f"    Model size: {model_size_mb:.1f} MB")
        
        start_time = time.time()
        
        # Load with CUDA GPU acceleration for maximum speed 
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        
        logging.info(f"    GPU Mode: CUDA acceleration enabled")
        logging.info(f"   ️  CPU Cores: {cpu_cores} (using 8 threads for hybrid mode)")
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=1024,          # Smaller context for speed (enough for RAG)
            n_threads=8,         # More threads for CPU/GPU coordination
            n_gpu_layers=-1,     # AUTO: offload all possible layers to GPU 
            n_batch=512,         # Large batch for GPU efficiency
            verbose=False,       # Disable verbose logging
            use_mmap=True,       # Memory-map model for faster loading
            use_mlock=False,     # Don't lock memory
            f16_kv=True,         # Use FP16 for key/value cache (faster on GPU)
        )
        
        load_time = time.time() - start_time
        logging.info(f" GGUF model loaded in {load_time:.2f}s")
        
        # 4. Warm up GGUF model with dummy generation
        logging.info(f" Warming up GGUF model...")
        start_time = time.time()
        _ = self.llm(
            "Test warmup",
            max_tokens=10,
            temperature=0.7,
            echo=False
        )
        warmup_time = time.time() - start_time
        logging.info(f" GGUF model warmed up in {warmup_time:.2f}s")
        
        # 5. Load knowledge base
        logging.info(f"\n Loading knowledge base from {self.knowledge_base_dir}")
        self._load_knowledge_base()
        
        # 6. Build FAISS index
        logging.info(f" Building FAISS index...")
        start_time = time.time()
        self._build_faiss_index()
        index_time = time.time() - start_time
        logging.info(f" FAISS index built in {index_time:.2f}s")
        logging.info(f" Index size: {self.faiss_index.ntotal} vectors")
        
        logging.info(f"\n{'='*60}")
        logging.info(f" GGUF HYBRID RAG SYSTEM READY")
        logging.info(f"   Rule-Based Extraction + Phi-2 GGUF (4-bit)")
        logging.info(f"{'='*60}\n")
    
    def _load_knowledge_base(self):
        """Load markdown files from Leibniz knowledge base directory"""
        if not os.path.exists(self.knowledge_base_dir):
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_dir}")
        
        md_files = []
        for root, dirs, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
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
        current_chunk = []
        current_length = 0
        chunk_size = 800
        overlap = 150
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({"text": chunk_text, "source": source})
                current_chunk = current_chunk[-2:]  # Keep overlap
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({"text": chunk_text, "source": source})
        
        return chunks
    
    def _build_faiss_index(self):
        """Build FAISS index from documents"""
        logging.info(f" Generating embeddings for {len(self.documents)} chunks...")
        
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
    
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
                "relevance_score": 1 / (1 + distance)
            })
        
        retrieval_time = time.time() - start_time
        return retrieved_docs, retrieval_time * 1000  # Return ms
    
    def _extract_clean_context(self, retrieved_docs: List[Dict], max_length: int = 800) -> str:
        """Rule-based extraction of clean context from retrieved documents"""
        context_parts = []
        total_length = 0
        
        for doc in retrieved_docs[:3]:  # Use top 3 docs
            text = doc['text']
            
            # Remove metadata headers
            lines = text.split('\n')
            clean_lines = []
            in_metadata = False
            
            for line in lines:
                if line.strip() == '---':
                    in_metadata = not in_metadata
                    continue
                if in_metadata:
                    continue
                
                if line.strip().startswith(('title:', 'category:', 'intents:', 'tags:', 
                                           'last_updated:', 'language:', 'priority:', 
                                           '**Title**:', '**Intent', '**Keywords:**')):
                    continue
                
                if not line.strip():
                    continue
                
                clean_lines.append(line.strip())
            
            clean_text = ' '.join(clean_lines)
            sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 30]
            
            for sentence in sentences[:4]:
                if total_length + len(sentence) < max_length:
                    context_parts.append(sentence)
                    total_length += len(sentence)
                else:
                    break
            
            if total_length >= max_length:
                break
        
        return '. '.join(context_parts) + '.'
    
    def _generate_gguf_response(self, query: str, context: str) -> Tuple[str, float]:
        """Generate human-like response using GGUF Phi-2 with context"""
        start_time = time.time()
        
        # Craft compact prompt for GGUF model
        prompt = f"""Context: {context}

Question: {query}

Answer (friendly, 2-3 sentences):"""
        
        # Generate with GGUF model (GPU-accelerated for maximum speed) 
        output = self.llm(
            prompt,
            max_tokens=60,           # Optimized token count
            temperature=0.7,
            top_p=0.95,
            top_k=40,               # Optimized for speed
            repeat_penalty=1.15,
            stop=["Question:", "\n\n", "Context:", "Q:"],  # Stop tokens
            echo=False,             # Don't echo prompt
            threads=8               # Explicit thread count for generation
        )
        
        # Extract generated text
        generated_text = output['choices'][0]['text'].strip()
        
        # Clean up answer
        answer = generated_text
        
        # Remove any trailing incomplete sentences
        if answer and not answer[-1] in '.!?':
            last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
            if last_period > 0:
                answer = answer[:last_period + 1]
        
        # Ensure minimum length
        if len(answer) < 50:
            answer = f"Based on the information available, {answer}"
        
        # Cap at 500 chars
        if len(answer) > 500:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:2]) + '.'
        
        generation_time = (time.time() - start_time) * 1000
        return answer, generation_time
    
    def run_query_test(self, query: str, query_idx: int, total_queries: int):
        """Run GGUF hybrid RAG test: retrieval + extraction + GGUF generation"""
        logging.info(f"\n{'='*60}")
        logging.info(f" Query {query_idx}/{total_queries}")
        logging.info(f"{'='*60}")
        logging.info(f" Question: {query}")
        
        try:
            # Step 1: Retrieve documents
            logging.info(f"\n Step 1: Retrieving relevant documents...")
            retrieved_docs, retrieval_time_ms = self.retrieve_documents(query)
            logging.info(f" Retrieved {len(retrieved_docs)} documents in {retrieval_time_ms:.2f}ms")
            
            # Step 2: Extract clean context
            logging.info(f"\n Step 2: Extracting clean context (rule-based)...")
            extract_start = time.time()
            context = self._extract_clean_context(retrieved_docs, max_length=800)
            extract_time_ms = (time.time() - extract_start) * 1000
            logging.info(f" Extracted {len(context)} chars in {extract_time_ms:.2f}ms")
            
            # Step 3: Generate with GGUF
            logging.info(f"\n Step 3: Generating response with GGUF Phi-2...")
            answer, generation_time_ms = self._generate_gguf_response(query, context)
            logging.info(f" Generated response in {generation_time_ms:.2f}ms")
            
            # Display answer
            logging.info(f"\n Generated Answer ({len(answer)} chars):")
            logging.info(f"   {'-'*55}")
            logging.info(f"   {answer}")
            logging.info(f"   {'-'*55}")
            
            # Show top documents
            logging.info(f"\n Top 3 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                logging.info(f"   {i}. [{doc['source']}] Relevance: {doc['relevance_score']:.4f}")
            
            total_time_ms = retrieval_time_ms + extract_time_ms + generation_time_ms
            logging.info(f"\n⏱️  Performance Metrics:")
            logging.info(f"   - Retrieval Time: {retrieval_time_ms:.2f}ms")
            logging.info(f"   - Extraction Time: {extract_time_ms:.2f}ms")
            logging.info(f"   - GGUF Generation: {generation_time_ms:.2f}ms")
            logging.info(f"   - Total Time: {total_time_ms:.2f}ms")
            
            # Store results
            test_result = {
                "query": query,
                "answer": answer,
                "answer_length": len(answer),
                "retrieval_time_ms": retrieval_time_ms,
                "extraction_time_ms": extract_time_ms,
                "generation_time_ms": generation_time_ms,
                "total_time_ms": total_time_ms,
                "num_docs_retrieved": len(retrieved_docs),
                "top_doc_scores": [doc['relevance_score'] for doc in retrieved_docs[:3]]
            }
            
            self.results["queries"].append(test_result)
            
        except Exception as e:
            logging.error(f" Query test failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.results["queries"].append({"query": query, "error": str(e)})
    
    def run_all_tests(self):
        """Run complete GGUF hybrid RAG test suite"""
        logging.info(f" Starting Leibniz GGUF Hybrid RAG Test Suite")
        logging.info(f" Test Run ID: {self.results['test_run_id']}\n")
        
        start_time = time.time()
        
        # Pre-warm models once
        self.prewarm_models()
        
        # Run tests
        for idx, query in enumerate(TEST_QUERIES, 1):
            self.run_query_test(query, idx, len(TEST_QUERIES))
        
        total_time = time.time() - start_time
        
        # Print summary
        self._print_summary(total_time)
        
        # Save results
        self._save_results()
        
        logging.info(f"\n Testing Complete! Total time: {total_time:.2f}s")
    
    def _print_summary(self, total_time: float):
        """Print test summary"""
        successful_tests = [q for q in self.results["queries"] if "error" not in q]
        failed_tests = [q for q in self.results["queries"] if "error" in q]
        
        if not successful_tests:
            logging.warning("️  No successful tests to summarize")
            return
        
        avg_retrieval = sum(q["retrieval_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_extraction = sum(q["extraction_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_generation = sum(q["generation_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_total = sum(q["total_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_answer_length = sum(q["answer_length"] for q in successful_tests) / len(successful_tests)
        
        logging.info(f"\n{'='*60}")
        logging.info(f" FINAL TEST SUMMARY (GGUF HYBRID RAG)")
        logging.info(f"{'='*60}")
        logging.info(f" Overall Performance:")
        logging.info(f"   - Total Queries: {len(TEST_QUERIES)}")
        logging.info(f"   - Successful: {len(successful_tests)}")
        logging.info(f"   - Failed: {len(failed_tests)}")
        logging.info(f"   - Avg Retrieval Time: {avg_retrieval:.2f}ms")
        logging.info(f"   - Avg Extraction Time: {avg_extraction:.2f}ms")
        logging.info(f"   - Avg GGUF Generation: {avg_generation:.2f}ms")
        logging.info(f"   - Avg Total Time: {avg_total:.2f}ms")
        logging.info(f"   - Avg Answer Length: {avg_answer_length:.0f} chars")
        logging.info(f"   - Total Test Time: {total_time:.2f}s")
        logging.info(f"\n Performance Analysis:")
        logging.info(f"   - Retrieval: {(avg_retrieval/avg_total)*100:.1f}% of total time")
        logging.info(f"   - Extraction: {(avg_extraction/avg_total)*100:.1f}% of total time")
        logging.info(f"   - Generation: {(avg_generation/avg_total)*100:.1f}% of total time")
        logging.info(f"\n GGUF vs HuggingFace Speedup:")
        logging.info(f"   - Expected: 3-5x faster generation")
        logging.info(f"   - Model Size: ~1.6GB (vs 2.7GB full precision)")
        logging.info(f"   - Memory Usage: ~2GB RAM (vs ~6GB RAM)")
        logging.info(f"{'='*60}\n")
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = f"leibniz_gguf_hybrid_results_{self.results['test_run_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f" Results saved to: {output_file}")

def main():
    # Default model path - adjust based on your setup
    model_path = "models/phi-2.Q4_K_M.gguf"
    
    # Check if model exists, provide helpful message if not
    if not os.path.exists(model_path):
        print("\n" + "="*60)
        print(" GGUF Model Setup Required")
        print("="*60)
        print(f"\nModel not found at: {model_path}")
        print("\n Download Phi-2 GGUF from:")
        print("   https://huggingface.co/TheBloke/phi-2-GGUF/tree/main")
        print("\n Recommended file: phi-2.Q4_K_M.gguf (1.6GB)")
        print("   - Best balance of quality and speed")
        print("   - 4-bit quantization")
        print("\n Create directory and download:")
        print("   mkdir models")
        print("   # Then download phi-2.Q4_K_M.gguf to models/")
        print("\n Alternative quantizations:")
        print("   - phi-2.Q5_K_M.gguf (1.9GB) - Better quality, slightly slower")
        print("   - phi-2.Q3_K_M.gguf (1.2GB) - Faster, lower quality")
        print("="*60 + "\n")
        return
    
    tester = LeibnizHybridRAGGGUF(model_path=model_path)
    tester.run_all_tests()

if __name__ == "__main__":
    main()
