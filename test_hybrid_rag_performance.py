"""
Test Hybrid RAG Performance - Pattern-Based + Standard RAG
===========================================================

Compares hybrid approach (rule-based + boosted retrieval) vs standard RAG
"""

import asyncio
import time
from leibniz_agent.leibniz_rag import LeibnizRAG

# Test queries covering different patterns
TEST_QUERIES = [
    # Pattern: office_hours
    {
        "query": "What are the office hours for the admissions department?",
        "expected_pattern": "office_hours",
        "context": {
            "user_goal": "asking about university information",
            "key_entities": {"department": "admissions"},
            "extracted_meaning": "office hours for admissions department",
            "return_timing": True
        }
    },
    # Pattern: contact_info
    {
        "query": "How can I contact the student services office?",
        "expected_pattern": "contact_info",
        "context": {
            "user_goal": "asking about university information",
            "key_entities": {"department": "student services"},
            "extracted_meaning": "contact information for student services",
            "return_timing": True
        }
    },
    # Pattern: admission_requirements
    {
        "query": "What are the admission requirements for the Master's program?",
        "expected_pattern": "admission_requirements",
        "context": {
            "user_goal": "asking about admission process",
            "key_entities": {"program": "master"},
            "extracted_meaning": "admission requirements for master's program",
            "return_timing": True
        }
    },
    # Pattern: appointment_scheduling
    {
        "query": "How do I schedule an appointment with an academic advisor?",
        "expected_pattern": "appointment_scheduling",
        "context": {
            "user_goal": "appointment scheduling",
            "key_entities": {"service": "academic advising"},
            "extracted_meaning": "schedule appointment with academic advisor",
            "return_timing": True
        }
    },
    # No pattern (fallback to standard RAG)
    {
        "query": "Tell me about research opportunities at Leibniz University",
        "expected_pattern": None,
        "context": {
            "user_goal": "asking about university information",
            "key_entities": {"topic": "research"},
            "extracted_meaning": "research opportunities at Leibniz University",
            "return_timing": True
        }
    }
]


def test_hybrid_rag():
    """Test hybrid RAG performance"""
    
    print("\n" + "="*70)
    print("🧪 HYBRID RAG PERFORMANCE TEST")
    print("="*70)
    
    # Initialize RAG system
    print("\n📥 Initializing Leibniz RAG...")
    rag = LeibnizRAG()
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(TEST_QUERIES)}")
        print(f"{'='*70}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Pattern: {test_case['expected_pattern'] or 'None (standard RAG)'}")
        print("-" * 70)
        
        try:
            # Process query
            result = rag.process_rag_query(
                context=test_case['context'],
                query=test_case['query']
            )
            
            # Unpack result
            if isinstance(result, tuple):
                answer, timing = result
            else:
                answer = result
                timing = {}
            
            # Display results
            print(f"\n💬 Answer ({len(answer)} chars):")
            print(f"   {answer}")
            
            print(f"\n⏱️  Performance Breakdown:")
            if timing:
                if 'pattern_detection_ms' in timing:
                    print(f"   Pattern Detection: {timing['pattern_detection_ms']:.1f}ms")
                if 'embedding_ms' in timing:
                    print(f"   Embedding:         {timing['embedding_ms']:.1f}ms")
                if 'search_ms' in timing:
                    print(f"   Search:            {timing['search_ms']:.1f}ms")
                if 'extraction_ms' in timing:
                    print(f"   Extraction:        {timing['extraction_ms']:.1f}ms")
                if 'generation_ms' in timing:
                    print(f"   Generation:        {timing['generation_ms']:.1f}ms")
                if 'total_ms' in timing:
                    print(f"   TOTAL:             {timing['total_ms']:.1f}ms")
                    
                    # Categorize performance
                    total_ms = timing['total_ms']
                    if total_ms < 600:
                        perf_label = "⚡ EXCELLENT (Hybrid)"
                    elif total_ms < 1200:
                        perf_label = "✅ GOOD"
                    elif total_ms < 2000:
                        perf_label = "⚠️  ACCEPTABLE"
                    else:
                        perf_label = "❌ SLOW (Standard RAG)"
                    
                    print(f"\n   Performance: {perf_label}")
            
            results.append({
                "query": test_case['query'],
                "expected_pattern": test_case['expected_pattern'],
                "answer_length": len(answer),
                "timing": timing
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    if results:
        # Calculate averages
        hybrid_results = [r for r in results if r['expected_pattern'] is not None]
        standard_results = [r for r in results if r['expected_pattern'] is None]
        
        if hybrid_results:
            avg_hybrid = sum(r['timing'].get('total_ms', 0) for r in hybrid_results) / len(hybrid_results)
            print(f"\n🔧 Hybrid Pattern Queries (n={len(hybrid_results)}):")
            print(f"   Average Latency: {avg_hybrid:.1f}ms")
            print(f"   Expected Range:  200-600ms")
            if avg_hybrid < 600:
                print(f"   ✅ Target achieved!")
            else:
                print(f"   ⚠️  Slower than expected")
        
        if standard_results:
            avg_standard = sum(r['timing'].get('total_ms', 0) for r in standard_results) / len(standard_results)
            print(f"\n📚 Standard RAG Queries (n={len(standard_results)}):")
            print(f"   Average Latency: {avg_standard:.1f}ms")
            print(f"   Expected Range:  600-2200ms")
        
        if hybrid_results and standard_results:
            speedup = avg_standard / avg_hybrid
            print(f"\n🏆 Performance Improvement:")
            print(f"   Speedup: {speedup:.2f}x faster")
            print(f"   Time Saved: {avg_standard - avg_hybrid:.1f}ms per query")
        
        print(f"\n{'='*70}")
        print("✅ Hybrid RAG Test Complete!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    test_hybrid_rag()
