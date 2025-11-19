"""
Leibniz RAG Hybrid System - ULTRA-FAST CPU Version
===================================================

Optimized for maximum speed on CPU-only systems using:
- Q3_K_M quantization (lightest, fastest)
- Minimal context window (512 tokens)
- Aggressive token limits (40-50 tokens)
- Mirostat sampling for quality with fewer tokens
- Auto-tuned CPU threading

Expected Performance (CPU):
- Model Load: 2-3s
- Retrieval: 10-20ms
- Generation: 200-500ms 
- Total: ~600ms per query

Setup:
1. Download Q3 model (lightest): python download_phi2_gguf.py
   Select option 3: phi-2.Q3_K_M.gguf (1.2GB)
"""

import os
import sys
import logging
import time
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print(" llama-cpp-python not installed!")
    print("Install: pip install llama-cpp-python")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')

TEST_QUERIES = [
    "Tell me about the top courses in computer science",
    "What are the admission requirements?",
    "How do I schedule an appointment?",
    "What are the office hours?",
    "Tell me about research opportunities"
]

class LeibnizFastRAG:
    def __init__(self, knowledge_base_dir="leibniz_knowledge_base", 
                 model_path="models/phi-2.Q3_K_M.gguf"):
        
        logging.info(" Leibniz ULTRA-FAST RAG (CPU Optimized)\n")
        
        if not os.path.exists(model_path):
            logging.error(f" Model not found: {model_path}")
            logging.error(f" Download Q3 model (fastest): python download_phi2_gguf.py")
            logging.error(f"   Select option 3: phi-2.Q3_K_M.gguf (1.2GB)")
            sys.exit(1)
        
        self.knowledge_base_dir = knowledge_base_dir
        self.model_path = model_path
        
        # Auto-detect optimal CPU configuration
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        self.n_threads = max(4, cpu_cores - 1)
        
        logging.info(f"️  CPU: {cpu_cores} cores, using {self.n_threads} threads")
        logging.info(f" Knowledge: {knowledge_base_dir}")
        logging.info(f" Model: {os.path.basename(model_path)}\n")
        
        # Initialize components
        self._load_embedding_model()
        self._load_llm_model()
        self._load_knowledge_base()
        self._build_faiss_index()
        
        self.results = []
    
    def _load_embedding_model(self):
        """Load sentence transformer"""
        logging.info(" Loading embedding model...")
        start = time.time()
        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        logging.info(f" Embedder loaded in {time.time()-start:.2f}s\n")
    
    def _load_llm_model(self):
        """Load GGUF model with CPU optimization"""
        logging.info(f" Loading GGUF model...")
        start = time.time()
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=512,               # Minimal context for speed 
            n_threads=self.n_threads,
            n_gpu_layers=0,          # CPU-only
            n_batch=128,             # Small batches for CPU
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            low_vram=True           # Optimize for low memory
        )
        
        load_time = time.time() - start
        logging.info(f" GGUF loaded in {load_time:.2f}s")
        
        # Quick warmup
        logging.info(f" Warming up...")
        start = time.time()
        self.llm("Test", max_tokens=5, temperature=0.7)
        logging.info(f" Ready in {time.time()-start:.2f}s\n")
    
    def _load_knowledge_base(self):
        """Load markdown files"""
        logging.info(f" Loading knowledge base...")
        self.documents = []
        
        if not os.path.exists(self.knowledge_base_dir):
            logging.error(f" Knowledge base not found: {self.knowledge_base_dir}")
            sys.exit(1)
        
        for root, dirs, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks = self._chunk_text(content, file)
                        self.documents.extend(chunks)
        
        logging.info(f" Loaded {len(self.documents)} chunks\n")
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Fast chunking"""
        chunks = []
        sentences = text.split('\n')
        current = []
        length = 0
        
        for sentence in sentences:
            if length + len(sentence) > 600 and current:
                chunks.append({"text": '\n'.join(current), "source": source})
                current = current[-1:]
                length = len(current[0]) if current else 0
            current.append(sentence)
            length += len(sentence)
        
        if current:
            chunks.append({"text": '\n'.join(current), "source": source})
        
        return chunks
    
    def _build_faiss_index(self):
        """Build FAISS index"""
        logging.info(" Building FAISS index...")
        start = time.time()
        
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=False)
        
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype('float32'))
        
        logging.info(f" Index built in {time.time()-start:.2f}s ({self.faiss_index.ntotal} vectors)\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[Dict], float]:
        """Fast retrieval"""
        start = time.time()
        
        query_emb = self.embedder.encode([query])
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), top_k)
        
        docs = []
        for idx, dist in zip(indices[0], distances[0]):
            docs.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "score": 1 / (1 + dist)
            })
        
        return docs, (time.time() - start) * 1000
    
    def generate(self, query: str, context: str) -> Tuple[str, float]:
        """Ultra-fast generation with mirostat"""
        start = time.time()
        
        # Compact prompt
        prompt = f"Context: {context[:400]}\n\nQ: {query}\n\nA:"
        
        # Generate with aggressive speed settings
        output = self.llm(
            prompt,
            max_tokens=50,          # Very short 
            temperature=0.8,
            top_p=0.95,
            top_k=30,              # Smaller for speed
            repeat_penalty=1.1,
            mirostat_mode=2,       # Quality with fewer tokens
            mirostat_tau=4.0,
            mirostat_eta=0.1,
            stop=["Q:", "\n\n"],
            echo=False
        )
        
        answer = output['choices'][0]['text'].strip()
        
        # Quick cleanup
        if not answer or len(answer) < 20:
            answer = context[:200].split('\n')[0]
        
        return answer, (time.time() - start) * 1000
    
    def query(self, question: str):
        """Run full query"""
        logging.info(f" {question}")
        
        # Retrieve
        docs, retrieval_ms = self.retrieve(question, top_k=3)
        context = "\n".join([d["text"][:200] for d in docs])
        
        # Generate
        answer, generation_ms = self.generate(question, context)
        
        total_ms = retrieval_ms + generation_ms
        
        logging.info(f" Answer ({len(answer)} chars):")
        logging.info(f"   {answer}")
        logging.info(f"⏱️  Retrieval: {retrieval_ms:.1f}ms | Generation: {generation_ms:.1f}ms | Total: {total_ms:.1f}ms\n")
        
        self.results.append({
            "query": question,
            "answer": answer,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "answer_len": len(answer)
        })
    
    def run_tests(self):
        """Run all test queries"""
        logging.info("="*60)
        logging.info(" RUNNING ULTRA-FAST TESTS")
        logging.info("="*60 + "\n")
        
        for query in TEST_QUERIES:
            self.query(query)
        
        # Summary
        logging.info("="*60)
        logging.info(" PERFORMANCE SUMMARY")
        logging.info("="*60)
        
        avg_ret = np.mean([r["retrieval_ms"] for r in self.results])
        avg_gen = np.mean([r["generation_ms"] for r in self.results])
        avg_total = np.mean([r["total_ms"] for r in self.results])
        avg_len = np.mean([r["answer_len"] for r in self.results])
        
        logging.info(f"Queries Tested: {len(self.results)}")
        logging.info(f"Avg Retrieval:  {avg_ret:.1f}ms")
        logging.info(f"Avg Generation: {avg_gen:.1f}ms ")
        logging.info(f"Avg Total:      {avg_total:.1f}ms")
        logging.info(f"Avg Length:     {avg_len:.0f} chars")
        logging.info("="*60)

if __name__ == "__main__":
    rag = LeibnizFastRAG()
    rag.run_tests()
