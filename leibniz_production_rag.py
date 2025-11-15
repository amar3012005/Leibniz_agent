"""
Leibniz Production RAG - Optimized HuggingFace Phi-2
=====================================================

FASTEST version based on benchmarks:
- HuggingFace Phi-2: 2,164ms avg generation
- GGUF Phi-2 Q4: 7,168ms avg generation
→ HuggingFace is 3.3x FASTER on this system!

This is the PRODUCTION version using HF Phi-2 optimized settings.
"""

import os
import time
import logging
import numpy as np
import faiss
import torch
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(message)s')

TEST_QUERIES = [
    "Tell me about the top courses in computer science and its marking scheme",
    "What are the admission requirements for the Master's program?",
    "How do I schedule an appointment with an academic advisor?",
    "What are the office hours for the admissions department?",
    "Tell me about the research opportunities at Leibniz University"
]

class LeibnizProductionRAG:
    """Production RAG system with optimized HuggingFace Phi-2"""
    
    def __init__(self, knowledge_base_dir="leibniz_knowledge_base"):
        logging.info("🚀 Leibniz Production RAG (Optimized HF Phi-2)\n")
        
        self.knowledge_base_dir = knowledge_base_dir
        self.top_k = 5
        self.results = []
        
        # Initialize all components
        self._load_embedding_model()
        self._load_phi2_model()
        self._load_knowledge_base()
        self._build_faiss_index()
        
        logging.info("\n✅ PRODUCTION RAG SYSTEM READY\n")
    
    def _load_embedding_model(self):
        """Load sentence transformer"""
        logging.info("📥 Loading embedding model...")
        start = time.time()
        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cuda"
        )
        logging.info(f"✅ Embedder loaded in {time.time()-start:.2f}s")
        
        # Warmup
        logging.info("🔥 Warming up embedder...")
        start = time.time()
        _ = self.embedder.encode(["warmup"], show_progress_bar=False)
        logging.info(f"✅ Warmed up in {time.time()-start:.2f}s\n")
    
    def _load_phi2_model(self):
        """Load HuggingFace Phi-2 with optimized settings"""
        logging.info("📥 Loading Phi-2 model (HuggingFace)...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )
        
        self.model.eval()  # Set to evaluation mode
        
        logging.info(f"✅ Phi-2 loaded in {time.time()-start:.2f}s")
        
        # Warmup
        logging.info("🔥 Warming up Phi-2...")
        start = time.time()
        inputs = self.tokenizer("Test warmup", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        logging.info(f"✅ Warmed up in {time.time()-start:.2f}s\n")
    
    def _load_knowledge_base(self):
        """Load markdown knowledge base"""
        logging.info("📚 Loading knowledge base...")
        self.documents = []
        
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            chunks = self._chunk_text(content, file)
                            self.documents.extend(chunks)
                    except Exception as e:
                        logging.warning(f"⚠️ Could not load {file}: {e}")
        
        logging.info(f"✅ Loaded {len(self.documents)} chunks\n")
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks"""
        chunks = []
        sentences = text.split('\n')
        current = []
        length = 0
        
        for sentence in sentences:
            if length + len(sentence) > 800 and current:
                chunks.append({"text": '\n'.join(current), "source": source})
                current = current[-2:]
                length = sum(len(s) for s in current)
            current.append(sentence)
            length += len(sentence)
        
        if current:
            chunks.append({"text": '\n'.join(current), "source": source})
        
        return chunks
    
    def _build_faiss_index(self):
        """Build FAISS vector index"""
        logging.info("🔨 Building FAISS index...")
        start = time.time()
        
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
        
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype('float32'))
        
        logging.info(f"✅ Index built in {time.time()-start:.2f}s ({self.faiss_index.ntotal} vectors)\n")
    
    def retrieve(self, query: str) -> Tuple[List[Dict], float]:
        """Retrieve relevant documents"""
        start = time.time()
        
        query_emb = self.embedder.encode([query], show_progress_bar=False)
        distances, indices = self.faiss_index.search(query_emb.astype('float32'), self.top_k)
        
        docs = []
        for idx, dist in zip(indices[0], distances[0]):
            docs.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "score": 1 / (1 + dist)
            })
        
        return docs, (time.time() - start) * 1000
    
    def _clean_context(self, docs: List[Dict]) -> str:
        """Extract clean context from retrieved documents"""
        context_parts = []
        
        for doc in docs[:3]:
            text = doc['text']
            lines = [l for l in text.split('\n') if l.strip() and 
                    not l.strip().startswith(('title:', 'category:', '---', '**Title**'))]
            clean_text = '\n'.join(lines[:15])  # First 15 lines
            if clean_text:
                context_parts.append(clean_text)
        
        return '\n\n'.join(context_parts)[:700]  # Max 700 chars
    
    def generate(self, query: str, context: str) -> Tuple[str, float]:
        """Generate answer with optimized Phi-2"""
        start = time.time()
        
        prompt = f"""Context: {context}

Question: {query}

Answer (friendly, 2-3 sentences):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                min_new_tokens=40,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Clean up answer
        if not answer or len(answer) < 50:
            answer = context.split('\n')[0][:200]
        
        if len(answer) > 500:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:3]) + '.'
        
        return answer, (time.time() - start) * 1000
    
    def query(self, question: str, idx: int = 0, total: int = 0):
        """Run full query"""
        logging.info("="*60)
        if total > 0:
            logging.info(f"🧪 Query {idx}/{total}")
        logging.info("="*60)
        logging.info(f"❓ {question}\n")
        
        # Retrieve
        docs, retrieval_ms = self.retrieve(question)
        context = self._clean_context(docs)
        
        # Generate
        answer, generation_ms = self.generate(question, context)
        
        total_ms = retrieval_ms + generation_ms
        
        logging.info(f"💬 Answer ({len(answer)} chars):")
        logging.info(f"   {answer}\n")
        logging.info(f"⏱️  Retrieval: {retrieval_ms:.1f}ms | Generation: {generation_ms:.1f}ms | Total: {total_ms:.1f}ms\n")
        
        self.results.append({
            "query": question,
            "answer": answer,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms
        })
    
    def run_tests(self):
        """Run test queries"""
        for i, query in enumerate(TEST_QUERIES, 1):
            self.query(query, i, len(TEST_QUERIES))
        
        # Summary
        logging.info("="*60)
        logging.info("📊 PERFORMANCE SUMMARY")
        logging.info("="*60)
        
        avg_ret = np.mean([r["retrieval_ms"] for r in self.results])
        avg_gen = np.mean([r["generation_ms"] for r in self.results])
        avg_total = np.mean([r["total_ms"] for r in self.results])
        
        logging.info(f"Queries: {len(self.results)}")
        logging.info(f"Avg Retrieval:  {avg_ret:.0f}ms")
        logging.info(f"Avg Generation: {avg_gen:.0f}ms")
        logging.info(f"Avg Total:      {avg_total:.0f}ms")
        logging.info("="*60)

if __name__ == "__main__":
    rag = LeibnizProductionRAG()
    rag.run_tests()
