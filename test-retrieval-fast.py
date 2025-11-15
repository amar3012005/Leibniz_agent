"""
Leibniz RAG Retrieval Test - Humanized Answers Mode
Retrieval + Simple answer generation (300-500 chars) with performance metrics
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

class LeibnizRAGRetrieval:
    def __init__(self, knowledge_base_dir="leibniz_knowledge_base"):
        logging.info(f"🚀 Initializing Leibniz RAG Retrieval (Humanized Mode)")
        logging.info(f"📁 Knowledge base: {knowledge_base_dir}\n")
        
        self.knowledge_base_dir = knowledge_base_dir
        self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.top_k = 5
        
        # Initialize storage
        self.embedder = None
        self.documents = []
        self.faiss_index = None
        
        # Results tracking
        self.results = {
            "test_run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "queries": []
        }
    
    def prewarm_models(self):
        """Pre-warm embedding model and build FAISS index"""
        logging.info(f"{'='*60}")
        logging.info(f"🔥 PRE-WARMING MODELS")
        logging.info(f"{'='*60}")
        
        # Load embedding model
        logging.info(f"📥 Loading embedding model: {self.embedding_model_name}")
        start_time = time.time()
        self.embedder = SentenceTransformer(self.embedding_model_name)
        embed_load_time = time.time() - start_time
        logging.info(f"✅ Embedding model loaded in {embed_load_time:.2f}s")
        
        # Warmup
        logging.info(f"🔥 Warming up embedding model...")
        start_time = time.time()
        _ = self.embedder.encode(["test query for warmup"], show_progress_bar=False)
        warmup_time = time.time() - start_time
        logging.info(f"✅ Embedding model warmed up in {warmup_time:.2f}s")
        
        # Load knowledge base
        logging.info(f"📚 Loading knowledge base from {self.knowledge_base_dir}")
        self._load_knowledge_base()
        
        # Build FAISS index
        logging.info(f"🔨 Building FAISS index...")
        start_time = time.time()
        self._build_faiss_index()
        index_time = time.time() - start_time
        logging.info(f"✅ FAISS index built in {index_time:.2f}s")
        logging.info(f"📊 Index size: {self.faiss_index.ntotal} vectors")
        
        logging.info(f"{'='*60}")
        logging.info(f"✅ RAG SYSTEM READY (HUMANIZED ANSWER MODE)")
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
                logging.warning(f"⚠️ Could not load {file_path}: {e}")
        
        logging.info(f"✅ Loaded {len(self.documents)} chunks from {len(md_files)} files")
    
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
        logging.info(f"🔢 Generating embeddings for {len(self.documents)} chunks...")
        
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
    
    def _generate_humanized_answer(self, query: str, retrieved_docs: List[Dict]) -> Tuple[str, float]:
        """Generate humanized 300-500 char answer from retrieved documents"""
        start_time = time.time()
        
        # Combine top 3 documents for context
        combined_text = "\n\n".join([doc['text'][:500] for doc in retrieved_docs[:3]])
        
        # Extract key information and create concise answer
        # Simple extraction logic - in production, use an LLM for better quality
        answer_parts = []
        
        # Try to extract most relevant sentences
        for doc in retrieved_docs[:2]:
            text = doc['text']
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            # Take first meaningful sentence
            for sentence in sentences[:2]:
                if len(sentence) > 30 and not sentence.startswith('---'):
                    answer_parts.append(sentence + '.')
                    break
        
        # Combine and truncate to 300-500 chars
        raw_answer = " ".join(answer_parts)
        
        # Ensure 300-500 char range
        if len(raw_answer) > 500:
            answer = raw_answer[:497] + "..."
        elif len(raw_answer) < 300:
            # Add more context from documents
            for doc in retrieved_docs[:3]:
                sentences = [s.strip() for s in doc['text'].split('.') if len(s.strip()) > 20]
                for sentence in sentences:
                    if len(sentence) > 30 and not sentence.startswith('---') and sentence not in answer_parts:
                        answer_parts.append(sentence + '.')
                        raw_answer = " ".join(answer_parts)
                        if len(raw_answer) >= 300:
                            break
                if len(raw_answer) >= 300:
                    break
            answer = raw_answer[:500] if len(raw_answer) > 500 else raw_answer
        else:
            answer = raw_answer
        
        generation_time_ms = (time.time() - start_time) * 1000
        return answer, generation_time_ms
    
    def run_query_test(self, query: str, query_idx: int, total_queries: int):
        """Run RAG retrieval test with humanized answer generation"""
        logging.info(f"\n{'='*60}")
        logging.info(f"🧪 Query {query_idx}/{total_queries}")
        logging.info(f"{'='*60}")
        logging.info(f"❓ Question: {query}")
        
        try:
            # Retrieve documents
            logging.info(f"\n🔍 Retrieving relevant documents...")
            retrieved_docs, retrieval_time_ms = self.retrieve_documents(query)
            logging.info(f"⚡ Retrieved {len(retrieved_docs)} documents in {retrieval_time_ms:.2f}ms")
            
            # Generate humanized answer
            logging.info(f"\n🤖 Generating humanized answer...")
            answer, generation_time_ms = self._generate_humanized_answer(query, retrieved_docs)
            
            # Display answer
            logging.info(f"\n💬 Generated Answer ({len(answer)} chars):")
            logging.info(f"   {'-'*55}")
            logging.info(f"   {answer}")
            logging.info(f"   {'-'*55}")
            
            # Show top documents used
            logging.info(f"\n📚 Top 3 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                logging.info(f"   {i}. [{doc['source']}] Relevance: {doc['relevance_score']:.4f}")
                preview = doc['text'][:150].replace('\n', ' ')
                logging.info(f"      {preview}...")
            
            total_time_ms = retrieval_time_ms + generation_time_ms
            logging.info(f"\n⏱️  Performance Metrics:")
            logging.info(f"   - Retrieval Time: {retrieval_time_ms:.2f}ms")
            logging.info(f"   - Generation Time: {generation_time_ms:.2f}ms")
            logging.info(f"   - Total Time: {total_time_ms:.2f}ms")
            
            # Store results
            test_result = {
                "query": query,
                "answer": answer,
                "answer_length": len(answer),
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
                "total_time_ms": total_time_ms,
                "num_docs_retrieved": len(retrieved_docs),
                "top_doc_scores": [doc['relevance_score'] for doc in retrieved_docs]
            }
            
            self.results["queries"].append(test_result)
            
        except Exception as e:
            logging.error(f"❌ Query test failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.results["queries"].append({"query": query, "error": str(e)})
    
    def run_all_tests(self):
        """Run complete RAG test suite"""
        logging.info(f"🎯 Starting Leibniz RAG Retrieval Test Suite")
        logging.info(f"📅 Test Run ID: {self.results['test_run_id']}\n")
        
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
        
        logging.info(f"\n✅ Testing Complete! Total time: {total_time:.2f}s")
    
    def _print_summary(self, total_time: float):
        """Print test summary"""
        successful_tests = [q for q in self.results["queries"] if "error" not in q]
        failed_tests = [q for q in self.results["queries"] if "error" in q]
        
        if not successful_tests:
            logging.warning("⚠️  No successful tests to summarize")
            return
        
        avg_retrieval = sum(q["retrieval_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_generation = sum(q["generation_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_total = sum(q["total_time_ms"] for q in successful_tests) / len(successful_tests)
        avg_answer_length = sum(q["answer_length"] for q in successful_tests) / len(successful_tests)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"🎉 FINAL TEST SUMMARY (HUMANIZED ANSWERS)")
        logging.info(f"{'='*60}")
        logging.info(f"📊 Overall Performance:")
        logging.info(f"   - Total Queries: {len(TEST_QUERIES)}")
        logging.info(f"   - Successful: {len(successful_tests)}")
        logging.info(f"   - Failed: {len(failed_tests)}")
        logging.info(f"   - Avg Retrieval Time: {avg_retrieval:.2f}ms")
        logging.info(f"   - Avg Generation Time: {avg_generation:.2f}ms")
        logging.info(f"   - Avg Total Time: {avg_total:.2f}ms")
        logging.info(f"   - Avg Answer Length: {avg_answer_length:.0f} chars")
        logging.info(f"   - Total Test Time: {total_time:.2f}s")
        logging.info(f"{'='*60}\n")
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = f"leibniz_retrieval_results_{self.results['test_run_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"💾 Results saved to: {output_file}")

def main():
    tester = LeibnizRAGRetrieval()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
