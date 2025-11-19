"""
Leibniz University Institute - Context-Aware RAG System

This module provides Retrieval-Augmented Generation (RAG) for the Leibniz University
customer service agent. It uses FAISS vector store with sentence-transformers embeddings
and Gemini 2.0 Flash for response generation.

Key Innovation - Context-Aware Retrieval:
Instead of accepting raw query strings, this RAG system accepts structured context from
the intent parser, including:
- user_goal: High-level description of what user wants
- key_entities: Extracted entities (program, department, topic, etc.)
- extracted_meaning: Normalized query for semantic search

This enables:
- Enhanced query embedding using normalized extracted_meaning
- Entity-based filtering to boost relevant documents
- Intent-aware response generation tailored to user goals

Knowledge Base:
- 63 markdown documents across 12 categories
- Categories: university_overview, faculties_departments, admission_enrollment,
  academic_programs, student_services, campus_facilities, academic_policies,
  administrative_procedures, research_opportunities, campus_life,
  location_transportation, contact_information

Architecture:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Vector Store: FAISS IndexFlatL2
- Response Generation: Gemini 2.0 Flash (169.5 tok/s)
- Chunking: Intelligent chunking with FAQ, section, and semantic strategies
- Tone: Friendly casual English (not formal academic)
"""

import os
import re  # ADDED for pattern extraction in hybrid approach
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
import numpy as np

# Disable CUDA and enable CPU-only mode for embeddings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import faiss
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from leibniz_agent.leibniz_config import get_leibniz_config, SEMANTIC_CONTEXT_DEBUG
from rag_cache import get_rag_cache_manager
import logging

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)


class LeibnizRAG:
    """Context-aware RAG system for Leibniz University customer service"""
    
    def __init__(self):
        """Initialize the Leibniz RAG system"""
        # Core components
        self.vector_store = None
        self.embeddings = None
        self.gemini_model = None
        
        # FIX: Resolve knowledge base path from repository root, not current working directory
        # Try LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH first, then LEIBNIZ_KNOWLEDGE_BASE_PATH, then default
        kb_path_raw = os.getenv("LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH") or os.getenv("LEIBNIZ_KNOWLEDGE_BASE_PATH") or "leibniz_knowledge_base"
        
        # FIX: If path is relative, resolve from repository root (not cwd)
        if os.path.isabs(kb_path_raw):
            self.knowledge_base_path = kb_path_raw
        else:
            # Compute repository root: parent of parent of this file
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.knowledge_base_path = os.path.join(repo_root, kb_path_raw)
        
        # Add fallback to alternative location if first path doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            # Try absolute path resolution as fallback
            fallback_path = os.path.abspath(kb_path_raw)
            if os.path.exists(fallback_path):
                self.knowledge_base_path = fallback_path
                print(f" Using fallback KB path: {self.knowledge_base_path}")
            else:
                print(f" WARNING: Knowledge base directory does not exist!")
                print(f"   Tried (repo root): {self.knowledge_base_path}")
                print(f"   Tried (cwd): {fallback_path}")
                print(f"   Set LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH to use absolute path")
        else:
            print(f" Knowledge base path (resolved from repo root): {self.knowledge_base_path}")
        
        # Count files to verify it's populated
        self.kb_missing = False
        self.kb_empty = False
        if os.path.exists(self.knowledge_base_path):
            md_count = sum(1 for root, dirs, files in os.walk(self.knowledge_base_path) for f in files if f.endswith('.md'))
            if md_count == 0:
                self.kb_empty = True
                print(f" WARNING: Knowledge base directory exists but contains no markdown files!")
                print(f"   Please populate with markdown files or run setup_simple_kb.py")
            else:
                print(f" Knowledge base exists ({md_count} markdown files found)")
        else:
            self.kb_missing = True
        
        self.vector_store_path = os.getenv(
            "LEIBNIZ_RAG_VECTOR_STORE_PATH",
            "leibniz_agent/vector_store"
        )
        
        # Embedding model (read from environment)
        self.embedding_model_name = os.getenv(
            "LEIBNIZ_RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Retrieval configuration (read from environment)
        self.top_k = int(os.getenv("LEIBNIZ_RAG_TOP_K", "8"))
        self.top_n = int(os.getenv("LEIBNIZ_RAG_TOP_N", "5"))
        self.similarity_threshold = float(os.getenv("LEIBNIZ_RAG_SIMILARITY_THRESHOLD", "0.3"))
        
        # Chunking configuration (read from environment)
        self.chunk_size_min = int(os.getenv("LEIBNIZ_RAG_CHUNK_SIZE_MIN", "500"))
        self.chunk_size_max = int(os.getenv("LEIBNIZ_RAG_CHUNK_SIZE_MAX", "800"))
        self.chunk_overlap = int(os.getenv("LEIBNIZ_RAG_CHUNK_OVERLAP", "100"))
        
        # Response generation configuration (read from environment)
        self.response_style = os.getenv("LEIBNIZ_RAG_RESPONSE_STYLE", "friendly_casual")
        self.max_response_length = int(os.getenv("LEIBNIZ_RAG_MAX_RESPONSE_LENGTH", "500"))
        self.enable_humanization = os.getenv("LEIBNIZ_RAG_ENABLE_HUMANIZATION", "true").lower() == "true"
        self.min_quality_score = float(os.getenv("LEIBNIZ_RAG_MIN_QUALITY_SCORE", "0.5"))
        
        # Performance settings (read from environment)
        self.enable_prewarm = os.getenv("LEIBNIZ_RAG_ENABLE_PREWARM", "true").lower() == "true"
        self.auto_build = os.getenv("LEIBNIZ_RAG_AUTO_BUILD", "true").lower() == "true"
        self.timeout = float(os.getenv("LEIBNIZ_RAG_TIMEOUT", "30.0"))
        
        # Document storage
        self.documents = []
        self.doc_metadata = []
        
        # Configuration
        self.leibniz_config = get_leibniz_config()
        
        # Initialize cache manager (singleton pattern)
        self.cache_manager = get_rag_cache_manager(namespace="leibniz")
        
        # HYBRID APPROACH: Rule-based patterns for instant context reduction
        # Reduces Gemini token count by 50-70% for common queries
        self.quick_answer_patterns = {
            "office_hours": {
                "keywords": ["office hours", "opening hours", "working hours", "open time", "office time", "hours of operation"],
                "response_template": "The {department} is open {hours}. {additional_info}",
                "faiss_boost": ["office_hours", "contact_information"],
                "max_context_chars": 2000,  # Comment 4: Increased from 400 to 2000 for richer context
                "priority": 1
            },
            "contact_info": {
                "keywords": ["contact", "email", "phone", "call", "reach", "get in touch", "contact information"],
                "response_template": "You can contact {department} via {contact_methods}.",
                "faiss_boost": ["contact_information"],
                "max_context_chars": 1500,  # Comment 4: Increased from 300 to 1500
                "priority": 1
            },
            "admission_requirements": {
                "keywords": ["admission", "requirements", "eligibility", "entry", "apply", "application"],
                "response_template": "For admission to {program}, you need: {requirements}.",
                "faiss_boost": ["admission", "masters_admission", "bachelors_admission"],
                "max_context_chars": 2500,  # Comment 4: Increased from 600 to 2500 (most benefit from context)
                "priority": 2
            },
            "appointment_scheduling": {
                "keywords": ["appointment", "schedule", "book", "meeting", "reservation", "slot"],
                "response_template": "To schedule an appointment: {steps}",
                "faiss_boost": ["appointment_scheduling", "administrative_procedures"],
                "max_context_chars": 1800,  # Comment 4: Increased from 400 to 1800
                "priority": 1
            },
            "tuition_fees": {
                "keywords": ["tuition", "fees", "cost", "price", "payment", "semester fee"],
                "response_template": "The {program} tuition is {amount}. {payment_info}",
                "faiss_boost": ["tuition", "fees", "financial"],
                "max_context_chars": 1800,  # Comment 4: Increased from 400 to 1800
                "priority": 2
            },
            "academic_calendar": {
                "keywords": ["semester", "calendar", "academic year", "term dates", "exam schedule"],
                "response_template": "The academic calendar shows: {calendar_info}",
                "faiss_boost": ["academic_calendar", "academic_policies"],
                "max_context_chars": 2200,  # Comment 4: Increased from 500 to 2200
                "priority": 2
            }
        }
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_gemini()
        self._load_vector_store()
        
        # Comment 1.2: Auto-build vector store at initialization if auto_build=True
        # This ensures FAISS index is ready immediately, not on first query
        if self.auto_build and self.vector_store is None:
            if not self.kb_missing and not self.kb_empty:
                print(" Auto-building vector store at initialization (auto_build=True)...")
                build_success = self._build_vector_store_from_knowledge_base()
                if build_success:
                    print(f" Vector store built successfully at initialization ({len(self.documents)} chunks)")
                else:
                    print(" Vector store auto-build failed - will retry on first query")
            else:
                print(" Skipping auto-build: Knowledge base missing or empty")
        
        print(" Leibniz RAG: Initialized for university customer service")
        print(f"   Knowledge base: {self.knowledge_base_path}")
        print(f"   Top-K: {self.top_k}, Top-N: {self.top_n}, Similarity threshold: {self.similarity_threshold}")
        print(f"    Hybrid mode: {len(self.quick_answer_patterns)} rule-based patterns loaded")

    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f" Embeddings model loaded successfully ({self.embedding_model_name})")
        except Exception as e:
            print(f" Failed to load embeddings model: {e}")
            print(" Using Gemini-only mode (embeddings will be generated via API)")
            self.embeddings = None
    
    def _initialize_gemini(self):
        """Initialize Gemini 2.0 Flash for response generation"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print(" GEMINI_API_KEY not found in environment")
                self.gemini_model = None
                return
            
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
            self.gemini_model = genai.GenerativeModel(model_name)
            print(f" Gemini 2.0 Flash initialized for Leibniz RAG (169.5 tok/s)")
        except Exception as e:
            print(f" Failed to initialize Gemini: {e}")
            self.gemini_model = None
    
    def _load_vector_store(self):
        """Load existing vector store or prepare to build"""
        try:
            index_path = os.path.join(self.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.vector_store_path, "texts.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(texts_path):
                # Load existing vector store
                self.vector_store = faiss.read_index(index_path)
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.doc_metadata = json.load(f)
                
                with open(texts_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                
                print(f" Leibniz vector store loaded successfully ({len(self.documents)} chunks)")
            else:
                print(" Vector store not found, will build from leibniz_knowledge_base/")
                self.vector_store = None
        except Exception as e:
            print(f" Error loading vector store: {e}")
            self.vector_store = None
    
    def _build_vector_store_from_knowledge_base(self) -> bool:
        """Build vector store from Leibniz knowledge base"""
        try:
            return self._build_optimized_vector_store()
        except Exception as e:
            print(f" Optimized build failed: {e}, trying basic method")
            return self._build_basic_vector_store()
    
    def _build_optimized_vector_store(self) -> bool:
        """Build vector store with intelligent chunking"""
        try:
            # Comment 6: Abort if knowledge base path doesn't exist
            if not os.path.exists(self.knowledge_base_path):
                print(f" Knowledge base not found: {self.knowledge_base_path}")
                print(f"   Resolved absolute path: {os.path.abspath(self.knowledge_base_path)}")
                print(f"   Please create this directory and add markdown files,")
                print(f"   or run setup_simple_kb.py to generate sample content.")
                print(f"   Aborting vector store build to prevent zero-document index.")
                return False
            
            # Collect all markdown files from 12 category subdirectories
            all_documents = []
            all_metadata = []
            
            print(f" Scanning Leibniz knowledge base: {self.knowledge_base_path}")
            
            for root, dirs, files in os.walk(self.knowledge_base_path):
                for file in files:
                    if file.endswith('.md') and not file.startswith('00-'):
                        filepath = os.path.join(root, file)
                        
                        # Read file content
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Comment 6: Extract category from directory name, skip root files
                        category = os.path.basename(root)
                        # Skip files in root directory (no numbered category prefix)
                        if not re.match(r'^\d{2}_', category):
                            # Check if we're in root directory
                            if root == self.knowledge_base_path:
                                continue  # Skip root-level files like 00-README.md
                            # Otherwise, it's a valid category without numeric prefix
                        
                        # Apply intelligent chunking
                        chunks = self._intelligent_chunk_text(content, file)
                        
                        # Store chunks with metadata
                        for idx, chunk in enumerate(chunks):
                            all_documents.append(chunk)
                            all_metadata.append({
                                'source': file,
                                'category': category,
                                'chunk_id': idx,
                                'priority': self._get_content_priority(chunk, file, category)
                            })
            
            if not all_documents:
                # FIX: Consolidated clear error message with actionable fix
                print(" RAG system not ready: No markdown files found in knowledge base")
                print(f"   Expected location: {self.knowledge_base_path}")
                print(f"   Please verify the knowledge base exists at the repository root")
                print(f"   Or set LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH environment variable to absolute path")
                return False
            
            print(f" Processed {len(all_documents)} document chunks from {len(set(m['source'] for m in all_metadata))} files")
            
            # Create embeddings
            print(" Creating embeddings...")
            embeddings_array = self.embeddings.embed_documents(all_documents)
            embeddings_array = np.array(embeddings_array, dtype=np.float32)
            
            # Build FAISS index
            print(" Building FAISS index...")
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save to disk
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            index_path = os.path.join(self.vector_store_path, "index.faiss")
            metadata_path = os.path.join(self.vector_store_path, "metadata.json")
            texts_path = os.path.join(self.vector_store_path, "texts.json")
            
            faiss.write_index(index, index_path)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            
            # Update instance variables
            self.vector_store = index
            self.documents = all_documents
            self.doc_metadata = all_metadata
            
            print(f" Leibniz vector store built with {len(all_documents)} document chunks")
            return True
            
        except Exception as e:
            print(f" Error building optimized vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_basic_vector_store(self) -> bool:
        """Fallback method with simpler chunking"""
        try:
            all_documents = []
            all_metadata = []
            
            for root, dirs, files in os.walk(self.knowledge_base_path):
                for file in files:
                    if file.endswith('.md') and not file.startswith('00-'):
                        filepath = os.path.join(root, file)
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Comment 6: Extract category from directory name, skip root files
                        category = os.path.basename(root)
                        # Skip files in root directory (no numbered category prefix)
                        if not re.match(r'^\d{2}_', category):
                            # Check if we're in root directory
                            if root == self.knowledge_base_path:
                                continue  # Skip root-level files
                            # Otherwise, it's a valid category without numeric prefix
                        
                        # Simple chunking
                        chunks = self._split_into_chunks(content, chunk_size=800, overlap=100)
                        
                        for idx, chunk in enumerate(chunks):
                            all_documents.append(chunk)
                            all_metadata.append({
                                'source': file,
                                'category': category,
                                'chunk_id': idx,
                                'priority': 0
                            })
            
            if not all_documents:
                return False
            
            embeddings_array = self.embeddings.embed_documents(all_documents)
            embeddings_array = np.array(embeddings_array, dtype=np.float32)
            
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            faiss.write_index(index, os.path.join(self.vector_store_path, "index.faiss"))
            
            with open(os.path.join(self.vector_store_path, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False)
            
            with open(os.path.join(self.vector_store_path, "texts.json"), 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False)
            
            self.vector_store = index
            self.documents = all_documents
            self.doc_metadata = all_metadata
            
            print(f" Basic vector store built and saved ({len(all_documents)} chunks)")
            return True
            
        except Exception as e:
            print(f" Error building basic vector store: {e}")
            return False
    
    def _intelligent_chunk_text(self, content: str, filename: str) -> List[str]:
        """Apply different chunking strategies based on file type"""
        # FAQ files - split by Q&A pairs
        if 'faq' in filename.lower() or 'frequently' in filename.lower():
            return self._split_qa_content(content)
        
        # Guide/procedure files - split by sections
        if any(term in filename.lower() for term in ['guide', 'process', 'procedure', 'enrollment', 'admission']):
            return self._split_by_sections(content)
        
        # Default - semantic chunking
        return self._split_text_semantically(content)
    
    def _split_qa_content(self, content: str) -> List[str]:
        """Split FAQ content by question-answer pairs"""
        chunks = []
        
        # English Q&A patterns
        qa_patterns = [
            r'\n\s*[Qq]\d*[\.\):]\s+',
            r'\n\s*Question\s*[\d\.\):]*\s*',
            r'\n\s*FAQ\s*[\d\.\):]*\s*'
        ]
        
        # Split by Q&A patterns
        current_chunk = ""
        lines = content.split('\n')
        
        for line in lines:
            is_question = any(re.match(pattern, '\n' + line) for pattern in qa_patterns)
            
            if is_question and current_chunk:
                # Save previous Q&A pair
                if 50 < len(current_chunk.strip()) < 800:
                    chunks.append(current_chunk.strip())
                elif len(current_chunk.strip()) >= 800:
                    # Split oversized chunk with overlap
                    sub_chunks = self._split_into_chunks(current_chunk, chunk_size=800, overlap=100)
                    chunks.extend(sub_chunks)
                else:
                    # Too short, continue accumulating
                    pass
                
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        # Add final chunk
        if current_chunk.strip():
            if 50 < len(current_chunk.strip()) < 800:
                chunks.append(current_chunk.strip())
            elif len(current_chunk.strip()) >= self.chunk_size_max:
                sub_chunks = self._split_into_chunks(current_chunk, chunk_size=self.chunk_size_max, overlap=self.chunk_overlap)
                chunks.extend(sub_chunks)
        
        # If no chunks created, fallback to semantic splitting
        if not chunks:
            return self._split_text_semantically(content)
        
        return chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by markdown headers"""
        chunks = []
        
        # Split by headers (## or ###)
        sections = re.split(r'\n\s*#{1,6}\s+', content)
        
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If adding this section would exceed max size, save current chunk
            if len(current_chunk) + len(section) > self.chunk_size_max and current_chunk:
                if len(current_chunk) >= self.chunk_size_min:
                    chunks.append(current_chunk.strip())
                    current_chunk = section
                else:
                    # Current chunk too small, continue accumulating
                    current_chunk += "\n\n" + section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Add final chunk
        if current_chunk.strip():
            if len(current_chunk) >= self.chunk_size_min:
                chunks.append(current_chunk.strip())
            elif chunks:
                # Add to last chunk if too small
                chunks[-1] += "\n\n" + current_chunk.strip()
        
        # If no chunks created or only one small chunk, fallback
        if not chunks or (len(chunks) == 1 and len(chunks[0]) < self.chunk_size_min):
            return self._split_text_semantically(content)
        
        return chunks
    
    def _split_text_semantically(self, content: str) -> List[str]:
        """Split text by paragraphs with size constraints"""
        chunks = []
        
        # Split by paragraphs
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is very large, split it
            if len(para) > self.chunk_size_max:
                # Save current chunk if any
                if current_chunk and len(current_chunk) >= self.chunk_size_min:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split large paragraph
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend(sub_chunks)
            
            # If adding this paragraph would exceed max size
            elif len(current_chunk) + len(para) > self.chunk_size_max:
                if len(current_chunk) >= self.chunk_size_min:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    # Current chunk too small, continue accumulating
                    current_chunk += "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk) >= self.chunk_size_min:
            chunks.append(current_chunk.strip())
        elif current_chunk.strip() and chunks:
            # Add to last chunk
            chunks[-1] += "\n\n" + current_chunk.strip()
        
        return chunks if chunks else [content]
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraph by sentences"""
        chunks = []
        
        # Split by sentences (English period)
        sentences = paragraph.split('. ')
        
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back (except for last sentence which may have it)
            if i < len(sentences) - 1:
                sentence += '.'
            
            if len(current_chunk) + len(sentence) > self.chunk_size_max:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap (last sentence)
                    current_chunk = sentence
                else:
                    # Single sentence too long, force split
                    chunks.append(sentence[:self.chunk_size_max])
                    current_chunk = sentence[self.chunk_size_max - self.chunk_overlap:]  # overlap
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [paragraph]
    
    def _split_into_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Simple sentence-based chunking with overlap"""
        chunks = []
        
        # Split by sentences (English period)
        sentences = text.split('. ')
        
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add period back
            if i < len(sentences) - 1:
                sentence += '.'
            
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap by including last part of current chunk
                    if len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Handle final chunk
        if current_chunk.strip():
            # Avoid very small final chunk by merging with previous
            if len(current_chunk) < 200 and chunks:
                chunks[-1] += " " + current_chunk.strip()
            else:
                chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _get_content_priority(self, chunk: str, filename: str, category: str) -> int:
        """Assign priority scores to content chunks"""
        priority = 0
        
        # Higher priority for certain file types
        if any(term in filename.lower() for term in ['admission', 'enrollment', 'application']):
            priority += 10
        if 'faq' in filename.lower():
            priority += 8
        if any(term in filename.lower() for term in ['program', 'course', 'degree']):
            priority += 6
        
        # Higher priority for chunks with key terms
        university_terms = [
            'leibniz', 'admission', 'enrollment', 'program', 'course',
            'faculty', 'department', 'campus', 'student services', 'financial aid',
            'scholarship', 'tuition', 'housing', 'library'
        ]
        
        chunk_lower = chunk.lower()
        for term in university_terms:
            if term in chunk_lower:
                priority += 2
        
        # Higher priority for longer, more informative chunks
        if len(chunk) > 200:
            priority += 1
        
        return priority
    
    def _detect_query_pattern(self, query: str) -> Optional[Dict]:
        """
        Detect if query matches known patterns for hybrid retrieval (1-5ms)
        Comment 11: Enhanced with synonyms, fuzzy matching, and lower thresholds
        
        Args:
            query: User query text
            
        Returns:
            Pattern config dict with:
                - name: str
                - keywords: list
                - response_template: str
                - faiss_boost: list
                - max_context_chars: int
                - priority: int
            or None if no pattern match
        """
        import re
        query_lower = query.lower()
        
        # Comment 11: Add synonym mappings for fuzzy matching
        synonyms = {
            "office_hours": [
                "office hours", "opening hours", "working hours", "open time", "office time", 
                "hours of operation", "when open", "timing", "slot availability", "available hours",
                "office timing", "work hours", "working time"
            ],
            "contact_info": [
                "contact", "email", "phone", "call", "reach", "get in touch", "contact information",
                "contact details", "how to reach", "reach out", "speak to", "talk to"
            ],
            "admission_requirements": [
                "admission", "requirements", "eligibility", "entry", "apply", "application",
                "admission criteria", "entry requirements", "qualification", "prerequisites"
            ],
            "appointment_scheduling": [
                "appointment", "schedule", "book", "meeting", "reservation", "slot",
                "book slot", "make appointment", "reserve", "schedule meeting"
            ],
            "tuition_fees": [
                "tuition", "fees", "cost", "price", "payment", "semester fee",
                "tuition cost", "fee structure", "charges", "how much"
            ],
            "academic_calendar": [
                "semester", "calendar", "academic year", "term dates", "exam schedule",
                "semester dates", "academic schedule", "exam dates"
            ]
        }
        
        # Find best matching pattern (highest priority + fuzzy threshold)
        best_match = None
        best_priority = 0
        best_match_count = 0
        
        for pattern_name, pattern_config in self.quick_answer_patterns.items():
            # Comment 11: Use extended keywords from synonyms
            extended_keywords = synonyms.get(pattern_name, pattern_config["keywords"])
            
            # Comment 11: Count keyword matches for fuzzy threshold
            match_count = sum(1 for keyword in extended_keywords if keyword in query_lower)
            
            # Comment 11: Fuzzy regex patterns for hours/timing
            if pattern_name == "office_hours":
                # Check for time-related patterns
                time_pattern = r'\d{1,2}:\d{2}|\d{1,2}\s?(am|pm)|hours?|timing|schedule|open|available'
                if re.search(time_pattern, query_lower):
                    match_count += 1
            
            # Comment 11: Lower threshold - require at least 1 match (was implicit ANY)
            if match_count > 0:
                # Priority tie-breaker: more matches = better
                if (pattern_config.get("priority", 0) > best_priority or 
                    (pattern_config.get("priority", 0) == best_priority and match_count > best_match_count)):
                    best_match = {"name": pattern_name, **pattern_config}
                    best_priority = pattern_config.get("priority", 0)
                    best_match_count = match_count
        
        if best_match:
            logger.debug(f" Pattern detected: {best_match['name']} (match_count: {best_match_count})")
        
        return best_match
    
    def _retrieve_with_boosting(self, query: str, context: Dict, boost_categories: list, max_context_chars: int) -> tuple:
        """
        FAISS retrieval with category boosting and context truncation
        
        Args:
            query: User query text
            context: Structured context from intent parser
            boost_categories: List of categories/keywords to boost
            max_context_chars: Maximum total character count for context
            
        Returns:
            tuple: (relevant_docs: list, timing_dict: dict)
        """
        timing = {}
        
        # ENHANCED: Enrich query with entities AND specific keywords from query
        enriched_query = query
        
        # Extract specific department/office mentions from query
        dept_keywords = {
            "registrar": "registrar office registry student records enrollment verification transcript certificate exmatriculation",
            "admissions": "admissions office admission requirements application enroll apply eligibility",
            "academic": "academic affairs dean faculty program curriculum course",
            "student services": "student services counseling support advisory wellness",
            "international": "international office exchange study abroad visa foreign",
            "financial": "financial aid bafög scholarship tuition fees payment funding",
            "transcript": "transcript records registry registrar student services academic records",
            "certificate": "certificate certification registrar student services verification document",
        }
        
        # Add department-specific keywords if mentioned
        query_lower = query.lower()
        for dept, keywords in dept_keywords.items():
            if dept in query_lower:
                enriched_query = f"{query} {keywords}"
                logger.debug(f" Query enriched with department keywords: {dept}")
                break
        
        # Add entities from context
        if context and 'key_entities' in context:
            entities = context['key_entities']
            entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
            enriched_query = f"{enriched_query} {entity_terms}"
        
        # Embed query
        embed_start = time.time()
        query_embedding = self.embeddings.embed_query(enriched_query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        timing['embedding_ms'] = (time.time() - embed_start) * 1000
        
        # FAISS search (retrieve more candidates for filtering)
        search_start = time.time()
        distances, indices = self.vector_store.search(query_embedding, k=self.top_k + 5)
        timing['search_ms'] = (time.time() - search_start) * 1000
        
        # Build candidates with boosting
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                distance = float(distances[0][i])
                similarity = 1.0 - (distance * distance / 2.0)
                
                # Skip low-similarity docs
                if similarity < self.similarity_threshold:
                    continue
                
                doc_text = self.documents[idx]
                doc_meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                
                # Apply category boosting
                boosted_similarity = similarity
                doc_category = doc_meta.get('category', '').lower()
                doc_source = doc_meta.get('source', '').lower()
                
                # Check if doc matches any boost category
                for boost_cat in boost_categories:
                    boost_cat_lower = boost_cat.lower()
                    if boost_cat_lower in doc_category or boost_cat_lower in doc_source:
                        boosted_similarity *= 1.5  # 50% boost!
                        logger.debug(f" Boosted {doc_source} (matched '{boost_cat}')")
                        break  # Only boost once
                
                candidates.append({
                    'text': doc_text,
                    'metadata': doc_meta,
                    'distance': distance,
                    'similarity': similarity,
                    'boosted_similarity': boosted_similarity
                })
        
        # Sort by boosted similarity
        candidates.sort(key=lambda x: x['boosted_similarity'], reverse=True)
        
        # Truncate to max_context_chars (keeps highest-scoring docs)
        total_chars = 0
        final_docs = []
        
        for doc in candidates:
            doc_length = len(doc['text'])
            if total_chars + doc_length > max_context_chars:
                # Try to include partial doc if it fits
                remaining = max_context_chars - total_chars
                if remaining > 100:  # Only if meaningful chunk remains
                    doc_partial = {**doc, 'text': doc['text'][:remaining] + '...'}
                    final_docs.append(doc_partial)
                break
            
            final_docs.append(doc)
            total_chars += doc_length
        
        logger.info(f" Hybrid retrieval: {len(final_docs)} docs ({total_chars} chars, limit: {max_context_chars})")
        
        return final_docs, timing
    
    def _extract_template_fields(self, docs: list, pattern: Dict) -> Dict:
        """
        Extract structured fields from retrieved docs for template filling
        
        Args:
            docs: Retrieved document dicts
            pattern: Pattern configuration dict
            
        Returns:
            dict: Extracted fields for template (e.g., {department, hours, ...})
        """
        extracted = {}
        pattern_name = pattern.get("name", "")
        
        # Combine all doc texts for extraction
        combined_text = "\n".join([doc.get('text', '') for doc in docs])
        
        # Pattern-specific extraction logic
        if pattern_name == "office_hours":
            # Extract hours pattern (e.g., "Monday-Friday 9:00-15:00")
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Find department mention
            dept_keywords = ["admissions", "academic", "student services", "registry", "enrollment"]
            for dept in dept_keywords:
                if dept in combined_text.lower():
                    extracted["department"] = dept.title()
                    break
            
            # Extract additional info (walk-in, appointment, etc.)
            if "walk-in" in combined_text.lower():
                walk_in_match = re.search(r'walk-in.*?(\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
                if walk_in_match:
                    extracted["additional_info"] = f"Walk-in hours: {walk_in_match.group(1)}"
        
        elif pattern_name == "admission_requirements":
            # ENHANCED: More specific extraction for admission requirements
            # Extract program mention (with context)
            programs = {
                "master": ["master", "master's", "msc", "m.sc"],
                "bachelor": ["bachelor", "bachelor's", "bsc", "b.sc"],
                "phd": ["phd", "doctoral", "doctorate"],
                "mba": ["mba", "business administration"]
            }
            for prog_key, prog_variants in programs.items():
                if any(variant in combined_text.lower() for variant in prog_variants):
                    extracted["program"] = prog_key.upper() if prog_key == "mba" else prog_key.title()
                    break
            
            # Extract specific requirements with categories
            requirements_found = []
            
            # Check for GPA/grades
            gpa_match = re.search(r'(GPA|grade|average).*?(\d+\.?\d*)', combined_text, re.IGNORECASE)
            if gpa_match:
                requirements_found.append(f"GPA requirement: {gpa_match.group(2)}")
            
            # Check for language requirements
            if re.search(r'(TOEFL|IELTS|language|English|German)', combined_text, re.IGNORECASE):
                lang_match = re.search(r'(TOEFL.*?(?:\d+)|IELTS.*?(?:\d+\.?\d*))', combined_text, re.IGNORECASE)
                if lang_match:
                    requirements_found.append(f"Language: {lang_match.group(1)}")
                else:
                    requirements_found.append("Language proficiency required")
            
            # Check for degree requirements
            degree_match = re.search(r'(bachelor[\'s]?|undergraduate)\s+degree', combined_text, re.IGNORECASE)
            if degree_match:
                requirements_found.append(f"Previous {degree_match.group(1)} degree required")
            
            # Extract bulleted/numbered requirements
            req_lines = [line.strip() for line in combined_text.split('\n') 
                        if line.strip() and (line.strip().startswith(('-', '•', '*')) or re.match(r'^\d+\.', line.strip()))]
            if req_lines:
                # Filter out too-long lines (likely not actual requirements)
                clean_reqs = [r for r in req_lines if len(r) < 150][:3]
                requirements_found.extend(clean_reqs)
            
            if requirements_found:
                extracted["requirements"] = "; ".join(requirements_found[:4])  # Top 4 requirements
            
            # Extract application link/portal
            link_match = re.search(r'(https?://[^\s]+(?:application|apply|admission)[^\s]*)', combined_text, re.IGNORECASE)
            if link_match:
                extracted["application_link"] = link_match.group(1)
        
        elif pattern_name == "contact_info":
            # ENHANCED: More aggressive rule-based extraction for contact info
            # Extract ALL emails (prioritize .uni-hannover.de)
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
            uni_emails = [e for e in emails if 'uni-hannover' in e.lower() or 'leibniz' in e.lower()]
            if uni_emails:
                extracted["email"] = uni_emails[0]  # Prioritize university email
            elif emails:
                extracted["email"] = emails[0]
            
            # Extract ALL phones (prioritize German format)
            phones = re.findall(r'(\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4})', combined_text)
            if phones:
                extracted["phone"] = phones[0]
            
            # Extract office/department names
            dept_patterns = [
                r'(Registrar[\'s]?\s+Office)',
                r'(Student\s+Services)',
                r'(Academic\s+Affairs)',
                r'(Admissions\s+Office)',
                r'(International\s+Office)',
                r'(Central\s+Student\s+Advisory\s+Service)',
            ]
            for pattern_regex in dept_patterns:
                dept_match = re.search(pattern_regex, combined_text, re.IGNORECASE)
                if dept_match:
                    extracted["department"] = dept_match.group(1)
                    break
            
            # Extract office hours if mentioned
            hours_match = re.search(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)\s*[-–]\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)?\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})', combined_text, re.IGNORECASE)
            if hours_match:
                extracted["hours"] = hours_match.group(1)
            
            # Extract building/room numbers
            room_match = re.search(r'((?:Building|Room|Office)\s+[A-Z]?\d+[A-Za-z]?(?:\s*,\s*Room\s+\d+)?)', combined_text, re.IGNORECASE)
            if room_match:
                extracted["location"] = room_match.group(1)
            
            # Build contact summary
            contact_parts = []
            if "department" in extracted:
                contact_parts.append(f"the {extracted['department']}")
            if "email" in extracted:
                contact_parts.append(f"email: {extracted['email']}")
            if "phone" in extracted:
                contact_parts.append(f"phone: {extracted['phone']}")
            if "location" in extracted:
                contact_parts.append(f"location: {extracted['location']}")
            if "hours" in extracted:
                contact_parts.append(f"hours: {extracted['hours']}")
                
            if contact_parts:
                extracted["contact_summary"] = " | ".join(contact_parts)
        
        elif pattern_name == "appointment_scheduling":
            # Extract steps/instructions
            step_lines = [line.strip() for line in combined_text.split('\n') 
                         if line.strip() and (re.match(r'^\d+\.', line.strip()) or 'step' in line.lower())]
            if step_lines:
                extracted["steps"] = " ".join(step_lines[:3])
        
        return extracted
    
    def _build_hybrid_prompt(self, query: str, extracted_info: Dict, pattern: Dict, context: Optional[Dict], retrieved_docs: list) -> str:
        """
        Build SHORT prompt with template-based structure (reduces tokens by 50-70%)
        
        Args:
            query: User query
            extracted_info: Extracted template fields
            pattern: Pattern configuration
            context: Structured context from intent parser
            retrieved_docs: Retrieved documents
            
        Returns:
            str: Compact Gemini prompt
        """
        template = pattern["response_template"]
        pattern_name = pattern["name"]
        
        # Build compact context summary (only essential info)
        context_summary = ""
        if retrieved_docs:
            # Take first 300 chars of top doc (increased for better context)
            top_doc_text = retrieved_docs[0].get('text', '')[:300]
            context_summary = f"\nKey information: {top_doc_text}..."
        
        # ENHANCED: Pattern-specific prompt templates for better accuracy
        if pattern_name == "contact_info":
            # Force direct extraction-based response
            if extracted_info:
                prompt = f"""They asked: "{query}"

Here's the contact info you have:
{json.dumps(extracted_info, indent=2, ensure_ascii=False)}
{context_summary}

{"Respond in German." if context and context.get('language') == 'german' else ""}

Give them the contact information directly - email, phone, location, hours, whatever's relevant. Keep it short and clear (2-3 sentences). Keep your response between 150-200 characters total. Just give them what they need.

Your response:"""
            else:
                # Fallback if extraction failed
                prompt = f"""They asked: "{query}"
{context_summary}

{"Respond in German." if context and context.get('language') == 'german' else ""}

Give them the contact information from what you know. Be specific and direct. Keep your response between 150-200 characters total.

Your response:"""
        
        elif pattern_name == "admission_requirements":
            # Force specific, structured requirements response
            if extracted_info:
                prompt = f"""They asked: "{query}"

Requirements you found:
{json.dumps(extracted_info, indent=2, ensure_ascii=False)}
{context_summary}

{"Respond in German." if context and context.get('language') == 'german' else ""}

List out the admission requirements clearly. Mention the program if you know it. Keep it structured (3-4 sentences). Keep your response between 150-200 characters total. Only suggest contacting admissions if you're missing key info.

Your response:"""
            else:
                prompt = f"""They asked: "{query}"
{context_summary}

{"Respond in German." if context and context.get('language') == 'german' else ""}

Give them the admission requirements based on what you know. Be direct and clear. Keep your response between 150-200 characters total.

Your response:"""
        
        else:
            # Generic template for other patterns
            prompt = f"""They asked: "{query}"

Pattern type: {pattern_name.replace('_', ' ')}
Details: {json.dumps(extracted_info, ensure_ascii=False)}
{context_summary}

{"Respond in German." if context and context.get('language') == 'german' else ""}

Answer directly using the details you have. Be friendly and conversational (2-3 sentences). Keep your response between 150-200 characters total. If you don't have complete info, suggest who they should contact. Don't mention any template structure.

Your response:"""
        
        return prompt
    
    def process_rag_query(self, context: Dict[str, Any] = None, query: str = None, streaming_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        """
        Process RAG query with context-aware retrieval
        
        Args:
            context: Structured context from intent parser with:
                - user_goal: High-level intent description
                - key_entities: Extracted entities (program, department, topic, etc.)
                - extracted_meaning: Normalized query for semantic search
                - return_timing: Optional flag to return timing metrics
            query: Optional raw query (fallback if context not provided)
            streaming_callback: Optional callback for progressive text delivery
                - Signature: (partial_text: str, is_final: bool) -> None
                - Called with complete sentences as they are generated
            
        Returns:
            Generated response (str) or tuple of (response, timing_dict) if return_timing=True
        """
        start_time = time.time()
        timing = {}
        
        try:
            # Step 1: Extract query from context (NEW logic)
            query_text = ""
            if context:
                # Use extracted_meaning as primary query (normalized, semantic-rich)
                query_text = context.get('extracted_meaning', '')
                
                if query_text and SEMANTIC_CONTEXT_DEBUG:
                    logger.debug(f" Using extracted_meaning from context: '{query_text}'")
                
                # Fallback to user_goal if extracted_meaning empty
                if not query_text:
                    query_text = context.get('user_goal', '')
                    if query_text and SEMANTIC_CONTEXT_DEBUG:
                        logger.debug(f" Using user_goal from context: '{query_text}'")
            
            # Fallback to raw query parameter
            if not query_text:
                query_text = query if query else ""
                if query_text and SEMANTIC_CONTEXT_DEBUG:
                    logger.debug(f" No context available, using raw query: '{query_text}'")
            
            if not query_text:
                return "I'm sorry, I didn't understand your question. Could you please rephrase it?"
            
            # NEW: Check cache FIRST (before any processing)
            try:
                cached_result = self.cache_manager.get_query_response(query_text, language='mixed')
                if cached_result:
                    logger.info(f" Cache hit for query: '{query_text[:50]}...'")
                    return cached_result
            except Exception as cache_error:
                logger.warning(f" Cache check failed: {cache_error}")
                # Continue with normal processing
            
            # Step 2: Auto-build vector store if needed
            if self.vector_store is None:
                if SEMANTIC_CONTEXT_DEBUG:
                    logger.debug(" Building Leibniz vector store from knowledge base...")
                if not self._build_vector_store_from_knowledge_base():
                    return "I'm having trouble accessing the knowledge base. Please try again in a moment."
            
            # Step 3: Validate components
            if not self.embeddings:
                logger.warning(" Embeddings not available, using Gemini-only fallback with keyword filtering")
                # Gemini-only fallback: keyword-based document selection
                return_timing_flag = context.get('return_timing', False) if context else False
                return self._gemini_only_query(query_text, context, timing, return_timing_flag, streaming_callback)
            
            if not self.vector_store:
                logger.warning(" Vector store not available, using Gemini-only fallback")
                return_timing_flag = context.get('return_timing', False) if context else False
                return self._gemini_only_query(query_text, context, timing, return_timing_flag, streaming_callback)
            
            if not self.gemini_model:
                return "Response generation model not available. Please check API configuration."
            
            # HYBRID APPROACH: Check for pattern-based optimization (NEW STEP 3.5)
            pattern_start = time.time()
            detected_pattern = self._detect_query_pattern(query_text)
            timing['pattern_detection_ms'] = (time.time() - pattern_start) * 1000
            
            if detected_pattern:
                # HYBRID PATH: Use rule-based boosting + reduced context
                logger.info(f" Pattern detected: {detected_pattern['name']} (optimized hybrid path)")
                
                # Retrieve with category boosting and context truncation
                relevant_docs, retrieval_timing = self._retrieve_with_boosting(
                    query_text,
                    context,
                    boost_categories=detected_pattern.get("faiss_boost", []),
                    max_context_chars=detected_pattern.get("max_context_chars", 800)
                )
                timing.update(retrieval_timing)
                
                # Extract structured fields for template
                extract_start = time.time()
                extracted_info = self._extract_template_fields(relevant_docs, detected_pattern)
                timing['extraction_ms'] = (time.time() - extract_start) * 1000
                
                # Build compact hybrid prompt (50-70% fewer tokens)
                prompt = self._build_hybrid_prompt(
                    query_text,
                    extracted_info,
                    detected_pattern,
                    context,
                    relevant_docs
                )
                
                # Generate response with compact prompt
                gen_start = time.time()
                
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            top_p=0.9,
                            max_output_tokens=120,  # Comment 4 FIX: 100-130 for 150-200 char responses
                        )
                    )
                    answer = response.text.strip()
                    timing['generation_ms'] = (time.time() - gen_start) * 1000
                    
                    # NEW: Cache the successful hybrid result
                    try:
                        cache_metadata = {
                            'method': 'leibniz_rag_hybrid',
                            'pattern': detected_pattern.get('name', 'unknown'),
                            'relevance_score': 0.0,
                            'processing_time_ms': timing.get('total_ms', 0)
                        }
                        self.cache_manager.cache_query_response(query_text, answer, language='mixed', metadata=cache_metadata)
                        logger.debug(f" Cached Leibniz hybrid RAG response for query: '{query_text[:50]}...'")
                    except Exception as cache_error:
                        logger.warning(f" Failed to cache hybrid response: {cache_error}")
                    
                    # Return with timing if requested
                    total_time = (time.time() - start_time) * 1000
                    timing['total_ms'] = total_time
                    
                    return_timing_flag = context.get('return_timing', False) if context else False
                    if return_timing_flag:
                        return answer, timing
                    else:
                        return answer
                        
                except Exception as e:
                    logger.error(f" Hybrid generation failed: {e}")
                    # Fall through to standard RAG path
            
            # STANDARD RAG PATH: No pattern match or hybrid failed
            if detected_pattern:
                logger.info(" Falling back to standard RAG (hybrid failed)")
            else:
                logger.info(" Standard RAG path (no pattern match)")
            
            # Step 4: Enhanced query embedding (STANDARD logic)
            enriched_query = query_text
            
            # Add key entities to query for better matching
            if context and 'key_entities' in context:
                entities = context['key_entities']
                entity_terms = ' '.join([f"{k} {v}" for k, v in entities.items()])
                enriched_query = f"{query_text} {entity_terms}"
                if SEMANTIC_CONTEXT_DEBUG:
                    logger.debug(f" Query enriched with entities: '{enriched_query}'")
            
            # Add user goal for semantic context
            if context and 'user_goal' in context:
                user_goal = context['user_goal']
                enriched_query = f"{enriched_query} {user_goal}"
                if SEMANTIC_CONTEXT_DEBUG:
                    logger.debug(f" Query enriched with user_goal: final query length = {len(enriched_query)} chars")
            
            embed_start = time.time()
            query_embedding = self.embeddings.embed_query(enriched_query)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            timing['embedding_ms'] = (time.time() - embed_start) * 1000
            
            # Step 5: FAISS vector search with similarity threshold filtering
            search_start = time.time()
            distances, indices = self.vector_store.search(query_embedding, k=self.top_k)
            search_elapsed = (time.time() - search_start) * 1000
            timing['search_ms'] = search_elapsed
            
            # PHASE 2 CHANGE 2.4: Add retrieval timing log
            logger.info(f" Vector search completed in {search_elapsed:.1f}ms (top_k={self.top_k})")
            
            # Retrieve relevant documents and filter by similarity threshold
            relevant_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    # Convert L2 distance to cosine similarity (for normalized embeddings)
                    # similarity = 1 - (distance^2 / 2)
                    distance = float(distances[0][i])
                    similarity = 1.0 - (distance * distance / 2.0)
                    
                    # Apply similarity threshold filter
                    if similarity < self.similarity_threshold:
                        continue  # Skip documents below threshold
                    
                    doc_text = self.documents[idx]
                    doc_meta = self.doc_metadata[idx] if idx < len(self.doc_metadata) else {}
                    relevant_docs.append({
                        'text': doc_text,
                        'metadata': doc_meta,
                        'distance': distance,
                        'similarity': similarity
                    })
            
            # NEW: Entity-based filtering and boosting
            if context and 'key_entities' in context:
                entities = context['key_entities']
                
                # Boost documents from matching categories
                for doc in relevant_docs:
                    category = doc['metadata'].get('category', '')
                    
                    # Boost admission/enrollment docs if querying about programs or admission
                    if any(e in ['program', 'admission', 'enrollment'] for e in entities.keys()):
                        if 'admission' in category or 'enrollment' in category:
                            doc['priority_boost'] = 10
                        elif 'program' in category or 'academic' in category:
                            doc['priority_boost'] = 6
                        else:
                            doc['priority_boost'] = 0
                    
                    # Boost student services docs if querying about services
                    elif any(e in ['service', 'housing', 'financial aid'] for e in entities.keys()):
                        if 'student_services' in category:
                            doc['priority_boost'] = 10
                        else:
                            doc['priority_boost'] = 0
                    
                    # Boost course/program docs if querying about courses
                    elif any(e in ['course', 'class'] for e in entities.keys()):
                        if 'academic' in category or 'program' in category:
                            doc['priority_boost'] = 10
                        else:
                            doc['priority_boost'] = 0
                    
                    # Comment 5: Boost life/transport/contact docs (categories 10, 11, 12)
                    elif any(e in ['life', 'clubs', 'events', 'activities', 'social'] for e in entities.keys()):
                        if 'student_life' in category or '10_' in category:
                            doc['priority_boost'] = 10
                        else:
                            doc['priority_boost'] = 0
                    
                    elif any(e in ['transport', 'bus', 'parking', 'bike', 'public transit'] for e in entities.keys()):
                        if 'transport' in category or '11_' in category:
                            doc['priority_boost'] = 10
                        else:
                            doc['priority_boost'] = 0
                    
                    elif any(e in ['contact', 'office', 'emergency', 'help desk', 'support'] for e in entities.keys()):
                        if 'contact' in category or '12_' in category:
                            doc['priority_boost'] = 10
                        else:
                            doc['priority_boost'] = 0
                    
                    else:
                        doc['priority_boost'] = doc['metadata'].get('priority', 0)
                
                # Re-rank by priority boost + similarity (higher similarity = better)
                relevant_docs.sort(key=lambda x: (-x.get('priority_boost', 0), -x.get('similarity', 0)))
            
            # Select top N documents after filtering (use configured top_n)
            relevant_docs = relevant_docs[:self.top_n]
            
            # Step 6: Prepare context for Gemini
            context_text = "\n\n".join([doc['text'] for doc in relevant_docs])
            sources = list(set([doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]))
            
            # Step 7: Generate response with Leibniz-specific prompt
            user_goal_text = context.get('user_goal', 'general information') if context else 'general information'
            key_entities_text = str(context.get('key_entities', {})) if context else '{}'
            extracted_meaning = context.get('extracted_meaning', query_text) if context else query_text
            language = context.get('language', 'english') if context else 'english'
            
            prompt = f"""You're in the middle of a conversation as TARA, helping a student at Leibniz University.

They just asked: "{query_text}"

What you know:
{context_text}

{"Respond in German." if language == 'german' else ""}

Be direct and helpful - no need for greetings or formalities since you're already talking. Give them the information they need in 2-4 sentences. Keep your response between 150-200 characters total. If you don't have the answer in your knowledge base, be honest and point them to who can help. Sound natural, like you're actually talking to them.

Your response:"""
            
            # Step 8: Call Gemini API with streaming or standard generation
            gen_start = time.time()
            
            # Use streaming generation if callback provided
            if streaming_callback:
                accumulated_text = ""
                sentence_buffer = ""
                last_chunk_time = time.time()  # Comment 3: Track for timer-based flush (increased threshold)
                first_emit_done = False  # Comment 1: Track first emission for fast initial response
                sentinel_sent = False  # Comment 2 & 8 FIX: Track sentinel to send exactly once in finally
                
                try:
                    # Comment 8 FIX: Wrap streaming loop in try/finally to ensure sentinel in finally block
                    response_stream = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            top_p=0.9,
                            top_k=40,
                            max_output_tokens=120,  # Comment 4 FIX: 100-130 for 150-200 char responses
                        ),
                        stream=True
                    )
                    
                    # Process streaming chunks
                    for chunk in response_stream:
                        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                        accumulated_text += chunk_text
                        sentence_buffer += chunk_text
                        current_time = time.time()
                        
                        # Comment 3: Capture actual punctuation with regex capturing group
                        import re
                        sentence_pattern = r'([.!?])\s+'
                        split_result = re.split(sentence_pattern, sentence_buffer)
                        
                        # split_result format: [text, punct, text, punct, ..., remaining_text]
                        # Process complete sentences (pairs of text + punctuation)
                        if len(split_result) >= 3:  # At least one complete sentence
                            for i in range(0, len(split_result) - 2, 2):
                                sentence_text = split_result[i].strip()
                                punctuation = split_result[i + 1] if i + 1 < len(split_result) else '.'
                                
                                if sentence_text:
                                    # Comment 3: Re-emit with actual matched punctuation
                                    streaming_callback(sentence_text + punctuation, False)
                                    first_emit_done = True  # Comment 1: Mark first emission done
                            
                            # Keep incomplete fragment in buffer (last element)
                            sentence_buffer = split_result[-1] if split_result else ""
                            last_chunk_time = current_time
                        
                        # Comment 9: First-chunk fast emission - reduced thresholds for audible cue
                        # Emit first chunk at 300-400ms OR 15-20 chars (whichever comes first)
                        elif not first_emit_done:
                            time_elapsed_s = current_time - last_chunk_time
                            chars_available = len(sentence_buffer)
                            
                            # Whichever condition fires first
                            should_emit_time = time_elapsed_s >= 0.35  # 350ms fallback
                            should_emit_chars = chars_available >= 18  # 18 char threshold
                            
                            if should_emit_time or should_emit_chars:
                                # Fast flush for first chunk - emit what we have so far
                                first_chunk_text = sentence_buffer.strip()
                                if first_chunk_text:
                                    streaming_callback(first_chunk_text, False)
                                    
                                    # PHASE 2 CHANGE 2.5: Add first-sentence emission timing
                                    first_emit_elapsed = (time.time() - gen_start) * 1000
                                    logger.info(f" First chunk emitted in {first_emit_elapsed:.1f}ms: '{first_chunk_text[:40]}...' (trigger: {'time' if should_emit_time else 'chars'})")
                                    
                                    first_emit_done = True
                                    sentence_buffer = ""
                                    last_chunk_time = current_time
                        
                        # Comment 2: Adjusted clause-level timer flush - prefer complete sentences
                        # Increased timer threshold to 700ms and min-length to 70 chars
                        # Only flush if we have substantial content and significant pause
                        elif len(sentence_buffer) > 70:
                            clause_boundary_pattern = r'[,;]\s+'
                            clauses = re.split(clause_boundary_pattern, sentence_buffer)
                            
                            # Check if we have multiple clauses and enough time has passed (700ms)
                            if len(clauses) > 1 and (current_time - last_chunk_time) >= 0.7:
                                # Accumulate clauses until minimum 60 chars before emitting
                                accumulated_clause = ""
                                for clause in clauses[:-1]:
                                    accumulated_clause += clause.strip() + ", "
                                    if len(accumulated_clause) >= 60:
                                        # Emit accumulated clauses
                                        streaming_callback(accumulated_clause.strip(), False)
                                        logger.debug(f"⏱ Timer flush (700ms): '{accumulated_clause.strip()[:40]}...'")
                                        accumulated_clause = ""
                                
                                # Keep last incomplete clause in buffer
                                sentence_buffer = clauses[-1]
                                last_chunk_time = current_time
                    
                    # Comment 3: Send final fragment with original punctuation if available
                    if sentence_buffer.strip():
                        # Check if final fragment has trailing punctuation
                        final_text = sentence_buffer.strip()
                        if final_text and final_text[-1] not in '.!?':
                            # Add period if no punctuation
                            final_text += '.'
                        streaming_callback(final_text, False)  # Send final fragment as non-final
                    
                    raw_response = accumulated_text
                    
                except Exception as stream_error:
                    logger.error(f"Streaming generation error: {stream_error}")
                    # Fallback to non-streaming
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            top_p=0.9,
                            top_k=40,
                            max_output_tokens=120,  # Comment 4 FIX: 100-130 for 150-200 char responses
                        )
                    )
                    raw_response = response.text if response else "Sorry, I couldn't generate a response."
                
                finally:
                    # Comment 2 & 8 FIX: CRITICAL - Always send sentinel in finally block
                    # Ensures exactly one sentinel per stream, even on exceptions
                    if not sentinel_sent:
                        try:
                            streaming_callback("", True)  # Empty text with is_final=True acts as sentinel
                            sentinel_sent = True
                            logger.debug(" Sentinel sent from RAG streaming (finally block)")
                        except Exception as sentinel_error:
                            logger.error(f" Failed to send sentinel in finally: {sentinel_error}")
            else:
                # Standard non-streaming generation
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=120,  # Comment 4 FIX: 100-130 for 150-200 char responses
                    )
                )
                raw_response = response.text if response else "Sorry, I couldn't generate a response."
            
            timing['response_gen_ms'] = (time.time() - gen_start) * 1000
            
            # Step 9: Validate response quality
            quality = self._validate_response_quality(raw_response)
            
            if quality.get('retry', False) and quality.get('quality_score', 0) < 0.5:
                # Retry with modified prompt
                retry_prompt = prompt + "\n\nIMPORTANT: Please make the response more friendly and conversational, avoiding formal language."
                response = self.gemini_model.generate_content(retry_prompt)
                raw_response = response.text if response else raw_response
            
            # Comment 4 FIX: Trim response if > 200 chars (enforce 150-200 char limit)
            raw_response = self._trim_to_max_length(raw_response, max_chars=200)
            
            # Step 10: Humanize response for conversational English (skip for German)
            language = context.get('language', 'english') if context else 'english'
            if language == 'german':
                final_response = raw_response
            else:
                # Detect if this is first turn or follow-up based on context
                is_first_turn = True
                if context:
                    turn_number = context.get('turn_number', 1)
                    conversation_history = context.get('conversation_history', [])
                    is_first_turn = (turn_number <= 1 and len(conversation_history) == 0)
                
                final_response = self._humanize_response_english(raw_response, query_text, context, is_first_turn)
            
            # NEW: Cache the successful result
            try:
                cache_metadata = {
                    'method': 'leibniz_rag',
                    'relevance_score': 0.0,  # Could be enhanced with actual relevance scoring
                    'processing_time_ms': timing.get('total_ms', 0)
                }
                self.cache_manager.cache_query_response(query_text, final_response, language='mixed', metadata=cache_metadata)
                logger.debug(f" Cached Leibniz RAG response for query: '{query_text[:50]}...'")
            except Exception as cache_error:
                logger.warning(f" Failed to cache response: {cache_error}")
            
            timing['total_ms'] = (time.time() - start_time) * 1000
            
            # Step 11: Return response
            if context and context.get('return_timing'):
                return (final_response, timing)
            else:
                return final_response
            
        except Exception as e:
            print(f" Error processing RAG query: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I encountered an error while processing your question. Could you please try rephrasing it?"
    
    def _gemini_only_query(self, query_text: str, context: Dict[str, Any], timing: Dict[str, float], return_timing: bool, streaming_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        """Fallback query processing when embeddings/vector store unavailable - uses keyword filtering + Gemini"""
        try:
            # If no documents loaded at all, use Gemini without knowledge base
            if not self.documents:
                print(" No knowledge base documents available, using Gemini standalone mode")
                if not self.gemini_model:
                    return "I'm having trouble accessing the knowledge base. Please try again later."
                
                prompt = f"""You're in the middle of a conversation helping a student at Leibniz University.

They asked: {query_text}

You don't have specific knowledge base info right now, but answer as helpfully as you can based on general university knowledge. Be direct and friendly - no need for greetings since you're already talking. Keep your response between 150-200 characters total.

{"Respond in German." if context and context.get('language') == 'german' else ""}

Your response:"""
                
                try:
                    gen_start = time.time()
                    response = self.gemini_model.generate_content(prompt)
                    timing['generation_ms'] = (time.time() - gen_start) * 1000
                    
                    final_response = response.text.strip()
                    timing['total_ms'] = sum(timing.values())
                    
                    if return_timing:
                        return (final_response, timing)
                    return final_response
                except Exception as e:
                    print(f" Gemini generation error: {e}")
                    return "I'm having trouble generating a response. Please try again."
            
            # Keyword-based document selection from loaded documents
            keyword_start = time.time()
            keywords = query_text.lower().split()
            
            # Add entity keywords if available
            if context and 'key_entities' in context:
                for entity_value in context['key_entities'].values():
                    if isinstance(entity_value, str):
                        keywords.extend(entity_value.lower().split())
            
            # Score documents by keyword matches
            doc_scores = []
            for i, doc_text in enumerate(self.documents):
                doc_lower = doc_text.lower()
                score = sum(keyword in doc_lower for keyword in keywords)
                if score > 0:
                    doc_meta = self.doc_metadata[i] if i < len(self.doc_metadata) else {}
                    doc_scores.append({
                        'text': doc_text,
                        'metadata': doc_meta,
                        'score': score
                    })
            
            # Sort by score and select top N
            doc_scores.sort(key=lambda x: -x['score'])
            relevant_docs = doc_scores[:self.top_n]
            timing['keyword_search_ms'] = (time.time() - keyword_start) * 1000
            
            if not relevant_docs:
                # No keyword matches, use Gemini standalone
                print(" No keyword matches found, using Gemini standalone")
                prompt = f"""You're helping a student at Leibniz University right now.

They asked: {query_text}

You don't have specific documents on this topic, but do your best to help. Be direct and friendly. Keep your response between 150-200 characters total.

{"Respond in German." if context and context.get('language') == 'german' else ""}

Your response:"""
            else:
                # Generate response with matched documents
                context_text = "\n\n".join([doc['text'] for doc in relevant_docs])
                user_goal_text = context.get('user_goal', 'general information') if context else 'general information'
                
                prompt = f"""You're in the middle of a conversation helping a student at Leibniz University.

What you know:
{context_text}

They asked: {query_text}

Give them a helpful, direct answer using what you know above. Be natural and conversational - you're already talking to them, so skip the greetings. Keep your response between 150-200 characters total.

{"Respond in German." if context and context.get('language') == 'german' else ""}

Your response:"""
            
            # Generate response
            if not self.gemini_model:
                return "Response generation not available. Please check configuration."
            
            try:
                gen_start = time.time()
                response = self.gemini_model.generate_content(prompt)
                timing['generation_ms'] = (time.time() - gen_start) * 1000
                
                final_response = response.text.strip()
                
                # Apply humanization if enabled and not German
                language = context.get('language', 'english') if context else 'english'
                if self.enable_humanization and language != 'german':
                    final_response = self._humanize_response_english(final_response, query_text, context)
                
                timing['total_ms'] = sum(timing.values())
                
                if return_timing:
                    return (final_response, timing)
                return final_response
                
            except Exception as e:
                print(f" Gemini generation error: {e}")
                return "I encountered an error generating a response. Please try again."
                
        except Exception as e:
            print(f" Gemini-only query error: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I'm having trouble processing your question. Please try again."
    
    def _humanize_response_english(self, response: str, query: str, context: Dict[str, Any] = None, is_first_turn: bool = True) -> str:
        """Make responses more natural and conversational in English"""
        # Clean up response
        response = response.strip()
        
        # Remove formal/robotic prefixes that don't sound natural
        formal_prefixes = [
            "According to the context",
            "Based on the information provided",
            "As per the knowledge base",
            "The information indicates",
            "From the provided information",
            "Hey there",
            "Hi there", 
            "Hello",
            "Greetings",
            "Welcome",
            "I'm here to help",
            "How can I assist you",
            "Let me help you with that"
        ]
        
        for prefix in formal_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].lstrip(',.:; ')
        
        # Comment 4 FIX: Use CONCISE openers ("In short," / "Basically,")
        query_lower = query.lower()
        
        # Create stable hash for deterministic selection
        stable_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
        
        if not any(response.startswith(starter) for starter in ["In short", "Basically", "Here's", "So", "The", "You"]):
            if is_first_turn:
                # First turn: CONCISE starters only
                if query_lower.startswith('how'):
                    starters = ["In short,", "Basically,", "Here's how:"]
                    response = f"{starters[stable_hash % len(starters)]} {response[0].lower() + response[1:]}"
                elif query_lower.startswith('what'):
                    starters = ["Basically,", "In short,", "Here's the thing:"]
                    response = f"{starters[stable_hash % len(starters)]} {response[0].lower() + response[1:]}"
                elif query_lower.startswith('where'):
                    starters = ["In short,", "Basically,", "Here's where:"]
                    response = f"{starters[stable_hash % len(starters)]} {response[0].lower() + response[1:]}"
                elif query_lower.startswith('when'):
                    starters = ["In short,", "Basically,", "Here's when:"]
                    response = f"{starters[stable_hash % len(starters)]} {response[0].lower() + response[1:]}"
                elif query_lower.startswith('why'):
                    starters = ["In short,", "Basically,", "Here's why:"]
                    response = f"{starters[stable_hash % len(starters)]} {response[0].lower() + response[1:]}"
            else:
                # Follow-up turn: Even MORE concise
                starters = ["So", "Basically", "In short"]
                response = f"{starters[stable_hash % len(starters)]}, {response[0].lower() + response[1:]}"
        
        # Ensure proper sentence casing - first letter uppercase
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper sentence ending
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Comment 4 FIX: Enforce 200 char max (concise responses only)
        if len(response) > 200:
            response = self._trim_to_max_length(response, max_chars=200)
        
        return response
    
    def _trim_to_max_length(self, response: str, max_chars: int = 200) -> str:
        """
        Comment 4 FIX: Trim response to max_chars at sentence boundary
        
        Args:
            response: Raw response text
            max_chars: Maximum allowed characters (default 200 for 150-200 char limit)
        
        Returns:
            Trimmed response with ellipsis if truncated
        """
        if len(response) <= max_chars:
            return response
        
        # Find last sentence boundary before max_chars
        sentence_ends = ['.', '!', '?']
        best_cut = -1
        
        for end_char in sentence_ends:
            cut_pos = response.rfind(end_char, 0, max_chars)
            if cut_pos > best_cut and cut_pos > max_chars // 2:
                best_cut = cut_pos
        
        if best_cut > 0:
            # Cut at sentence boundary
            return response[:best_cut + 1]
        else:
            # No good boundary, hard cut with ellipsis
            return response[:max_chars - 3] + "..."
    
    def _validate_response_quality(self, response: str) -> Dict[str, Any]:
        """Validate generated response quality for friendly casual tone"""
        issues = []
        quality_score = 1.0
        
        # Check for overly formal language
        formal_words = ['pursuant', 'aforementioned', 'hereby', 'henceforth', 'notwithstanding']
        formal_count = sum(1 for word in formal_words if word in response.lower())
        if formal_count > 0:
            issues.append(f"Contains {formal_count} overly formal words")
            quality_score -= 0.2 * formal_count
        
        # Check for overly technical jargon
        jargon_words = ['matriculation', 'pedagogy', 'curriculum vitae']
        jargon_count = sum(1 for word in jargon_words if word in response.lower() and 'also known as' not in response.lower())
        if jargon_count > 0:
            issues.append(f"Contains {jargon_count} unexplained jargon")
            quality_score -= 0.15 * jargon_count
        
        # Check response length
        if len(response) < 30:
            issues.append("Response too short")
            quality_score -= 0.3
        elif len(response) > 200:
            issues.append("Response too long")
            quality_score -= 0.2
        
        # Check for helpfulness
        unhelpful_patterns = ["i don't know", "not sure", "can't help"]
        if any(pattern in response.lower() for pattern in unhelpful_patterns):
            if "but" not in response.lower() and "however" not in response.lower():
                issues.append("Unhelpful response without alternatives")
                quality_score -= 0.4
        
        # Determine if retry needed
        retry = quality_score < 0.5
        
        return {
            'issues': issues,
            'quality_score': max(0.0, quality_score),
            'retry': retry
        }
    
    def lightweight_prewarm(self):
        """Lightweight prewarm that accesses model objects without inference"""
        try:
            # Access embeddings model
            if self.embeddings:
                _ = self.embeddings.client
            
            # Access vector store
            if self.vector_store:
                _ = self.vector_store.ntotal
            
            # Access Gemini model
            if self.gemini_model:
                _ = self.gemini_model.model_name
            
            # Access documents
            _ = len(self.documents)
            
        except Exception:
            # Silent failure for non-critical operation
            pass
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about vector store"""
        stats = {
            'total_documents': len(self.documents),
            'vector_store_size': self.vector_store.ntotal if self.vector_store else 0,
            'knowledge_base_path': self.knowledge_base_path,
            'categories_loaded': len(set(meta.get('category', 'Unknown') for meta in self.doc_metadata))
        }
        return stats


# Global instance
_leibniz_rag_instance = None


def get_leibniz_rag():
    """Get the global Leibniz RAG instance (singleton pattern)"""
    global _leibniz_rag_instance
    if _leibniz_rag_instance is None:
        _leibniz_rag_instance = LeibnizRAG()
    return _leibniz_rag_instance


def process_leibniz_query(context: Dict[str, Any] = None, query: str = None) -> str:
    """
    Process RAG query using global Leibniz RAG instance
    
    Args:
        context: Structured context from intent parser
        query: Optional raw query (fallback)
        
    Returns:
        Generated response
    """
    rag = get_leibniz_rag()
    return rag.process_rag_query(context=context, query=query)


async def process_leibniz_query_async(context: Dict[str, Any] = None, query: str = None) -> str:
    """
    Async wrapper for RAG query processing
    
    Args:
        context: Structured context from intent parser
        query: Optional raw query (fallback)
        
    Returns:
        Generated response
    """
    # Run in thread to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, process_leibniz_query, context, query)


async def test_leibniz_rag():
    """Comprehensive test suite for Leibniz RAG"""
    print("\n" + "="*80)
    print("LEIBNIZ RAG TEST SUITE")
    print("="*80 + "\n")
    
    test_cases = [
        # Test with context from intent parser
        {
            "name": "CS Program Requirements (Context-Aware)",
            "context": {
                "user_goal": "asking about computer science program admission requirements",
                "key_entities": {"program": "computer science", "topic": "admission requirements"},
                "extracted_meaning": "computer science program admission requirements prerequisites",
                "return_timing": True
            },
            "expected_category": "admission_enrollment"
        },
        # Test with different entity types
        {
            "name": "Campus Housing (Context-Aware)",
            "context": {
                "user_goal": "inquiring about campus housing options and costs",
                "key_entities": {"service": "housing", "location": "campus", "topic": "cost"},
                "extracted_meaning": "campus housing availability cost options",
                "return_timing": True
            },
            "expected_category": "student_services"
        },
        # Test with raw query (fallback)
        {
            "name": "Library Hours (Raw Query)",
            "query": "What are the library hours?",
            "expected_category": "campus_facilities"
        },
        # Test with minimal context
        {
            "name": "Financial Aid (Minimal Context)",
            "context": {
                "user_goal": "asking about financial aid",
                "key_entities": {},
                "extracted_meaning": "financial aid scholarships"
            },
            "expected_category": "student_services"
        }
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    total_time = 0
    total_length = 0
    
    for idx, test in enumerate(test_cases, 1):
        print(f"Test {idx}: {test['name']}")
        print("-" * 80)
        
        # Get context or query
        context = test.get('context')
        query = test.get('query')
        
        # Print input
        if context:
            print(f"Context:")
            print(f"  User Goal: {context.get('user_goal', 'N/A')}")
            print(f"  Key Entities: {context.get('key_entities', {})}")
            print(f"  Extracted Meaning: {context.get('extracted_meaning', 'N/A')}")
        else:
            print(f"Query: {query}")
        
        # Process query
        result = process_leibniz_query(context=context, query=query)
        
        # Handle timing if returned
        if isinstance(result, tuple):
            response, timing = result
            print(f"\nTiming:")
            print(f"  Embedding: {timing.get('embedding_ms', 0):.2f}ms")
            print(f"  Search: {timing.get('search_ms', 0):.2f}ms")
            print(f"  Generation: {timing.get('response_gen_ms', 0):.2f}ms")
            print(f"  Total: {timing.get('total_ms', 0):.2f}ms")
            total_time += timing.get('total_ms', 0)
        else:
            response = result
        
        # Print response
        print(f"\nResponse ({len(response)} chars):")
        print(f"{response}")
        
        # Assess tone
        formal_words = ['pursuant', 'aforementioned', 'hereby']
        is_formal = any(word in response.lower() for word in formal_words)
        casual_phrases = ["here's", "basically", "you'll need to", "feel free"]
        is_casual = any(phrase in response.lower() for phrase in casual_phrases)
        
        tone = "Casual " if is_casual else ("Formal " if is_formal else "Neutral")
        print(f"Tone: {tone}")
        
        total_length += len(response)
        print()
    
    # Print summary
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Response Time: {total_time / len(test_cases):.2f}ms")
    print(f"Average Response Length: {total_length / len(test_cases):.0f} characters")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_leibniz_rag())
