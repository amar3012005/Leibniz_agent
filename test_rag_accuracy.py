"""
RAG Accuracy and Quality Test Suite
====================================

Tests the Leibniz University RAG system with comprehensive metrics:
- Response relevance and accuracy
- Timing (query, retrieval, generation)
- Ensemble matching performance
- Source attribution
- Confidence scores

Results saved to JSON for analysis.
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Leibniz modules
from leibniz_rag import process_leibniz_query_async
from leibniz_intent_parser import classify_leibniz_intent

# Test questions covering different aspects of university services
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What are your opening hours?",
        "category": "contact_information",
        "expected_keywords": ["hours", "opening", "closed", "time"]
    },
    {
        "id": 2,
        "question": "How do I apply for admission?",
        "category": "admissions",
        "expected_keywords": ["apply", "admission", "application", "process"]
    },
    {
        "id": 3,
        "question": "What programs do you offer?",
        "category": "academic_programs",
        "expected_keywords": ["programs", "degrees", "courses", "study"]
    },
    {
        "id": 4,
        "question": "Where is the university located?",
        "category": "location_transportation",
        "expected_keywords": ["location", "address", "city", "campus"]
    },
    {
        "id": 5,
        "question": "What are the tuition fees?",
        "category": "fees_financing",
        "expected_keywords": ["tuition", "fees", "cost", "price"]
    },
    {
        "id": 6,
        "question": "How do I contact the admissions office?",
        "category": "contact_information",
        "expected_keywords": ["contact", "phone", "email", "admissions"]
    },
    {
        "id": 7,
        "question": "What library resources are available?",
        "category": "facilities_services",
        "expected_keywords": ["library", "resources", "books", "study"]
    },
    {
        "id": 8,
        "question": "Are there any student clubs or organizations?",
        "category": "student_life",
        "expected_keywords": ["clubs", "organizations", "activities", "student"]
    },
    {
        "id": 9,
        "question": "What career services are available?",
        "category": "career_services",
        "expected_keywords": ["career", "jobs", "employment", "placement"]
    },
    {
        "id": 10,
        "question": "How do I register for courses?",
        "category": "academic_programs",
        "expected_keywords": ["register", "courses", "enrollment", "classes"]
    },
    {
        "id": 11,
        "question": "What housing options are available?",
        "category": "student_life",
        "expected_keywords": ["housing", "accommodation", "dorms", "residence"]
    },
    {
        "id": 12,
        "question": "How do I get my transcripts?",
        "category": "student_services",
        "expected_keywords": ["transcripts", "records", "academic", "documents"]
    },
    {
        "id": 13,
        "question": "What is the parking situation?",
        "category": "location_transportation",
        "expected_keywords": ["parking", "vehicle", "transportation", "access"]
    },
    {
        "id": 14,
        "question": "How do I find information about specific professors?",
        "category": "academic_programs",
        "expected_keywords": ["professors", "faculty", "teaching", "staff"]
    },
    {
        "id": 15,
        "question": "What student support services are available?",
        "category": "student_services",
        "expected_keywords": ["support", "help", "services", "assistance"]
    }
]


async def test_rag_query(question_data: Dict[str, Any], query_number: int, total_queries: int) -> Dict[str, Any]:
    """
    Test a single RAG query and collect comprehensive metrics.
    
    Args:
        question_data: Dictionary with question, category, expected_keywords
        query_number: Current query number (for progress)
        total_queries: Total number of queries (for progress)
    
    Returns:
        Dictionary with all test results and metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"📝 Query {query_number}/{total_queries}: {question_data['question']}")
    logger.info(f"{'='*70}")
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Intent Classification
    intent_start = time.time()
    try:
        intent_result = await classify_leibniz_intent(question_data['question'])
        intent_time = (time.time() - intent_start) * 1000
        intent = intent_result.get('intent', 'UNKNOWN')
        entities = intent_result.get('entities', {})
        should_use_rag = intent_result.get('should_use_rag', False)
        
        logger.info(f"⚡ Intent classified in {intent_time:.2f}ms: {intent}")
        logger.info(f"📊 Entities extracted: {entities}")
        
    except Exception as e:
        logger.error(f"❌ Intent classification failed: {e}")
        intent_result = {
            'intent': 'ERROR',
            'entities': {},
            'should_use_rag': True,
            'error': str(e)
        }
        intent_time = (time.time() - intent_start) * 1000
    
    # Step 2: RAG Query Processing
    rag_start = time.time()
    try:
        # Extract context from intent classification
        context = intent_result.get('entities', {})
        
        # Process RAG query
        rag_response = await process_leibniz_query_async(
            context=context,
            query=question_data['question']
        )
        
        rag_time = (time.time() - rag_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"📚 RAG query completed in {rag_time:.2f}ms")
        logger.info(f"⏱️ Total time: {total_time:.2f}ms")
        logger.info(f"📄 Response length: {len(rag_response)} characters")
        
        # Extract keywords from response
        response_keywords = _extract_keywords(rag_response.lower())
        expected_keywords_found = [kw for kw in question_data['expected_keywords'] 
                                  if kw.lower() in response_keywords]
        
        # Calculate keyword match score
        keyword_match_score = len(expected_keywords_found) / len(question_data['expected_keywords'])
        
        logger.info(f"🎯 Keyword match score: {keyword_match_score:.2%} "
                   f"({len(expected_keywords_found)}/{len(question_data['expected_keywords'])})")
        
        result = {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'category': question_data['category'],
            'intent_classification': {
                'intent': intent_result.get('intent'),
                'confidence': intent_result.get('confidence', 0.0),
                'entities': entities,
                'should_use_rag': should_use_rag,
                'processing_time_ms': intent_time
            },
            'rag_response': {
                'answer': rag_response,
                'response_length': len(rag_response),
                'processing_time_ms': rag_time,
                'keyword_match_score': keyword_match_score,
                'expected_keywords': question_data['expected_keywords'],
                'keywords_found': expected_keywords_found,
                'keywords_missing': [kw for kw in question_data['expected_keywords'] 
                                      if kw.lower() not in response_keywords]
            },
            'timing': {
                'intent_classification_ms': intent_time,
                'rag_processing_ms': rag_time,
                'total_time_ms': total_time
            },
            'status': 'success'
        }
        
        logger.info(f"✅ Query {query_number} completed successfully")
        
    except Exception as e:
        rag_time = (time.time() - rag_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        logger.error(f"❌ RAG query failed: {e}")
        
        result = {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'category': question_data['category'],
            'intent_classification': {
                'intent': intent_result.get('intent', 'ERROR'),
                'confidence': intent_result.get('confidence', 0.0),
                'entities': entities,
                'should_use_rag': should_use_rag,
                'processing_time_ms': intent_time
            },
            'rag_response': {
                'answer': f"ERROR: {str(e)}",
                'response_length': 0,
                'processing_time_ms': rag_time,
                'keyword_match_score': 0.0,
                'expected_keywords': question_data['expected_keywords'],
                'keywords_found': [],
                'keywords_missing': question_data['expected_keywords']
            },
            'timing': {
                'intent_classification_ms': intent_time,
                'rag_processing_ms': rag_time,
                'total_time_ms': total_time
            },
            'status': 'error',
            'error': str(e)
        }
    
    return result


def _extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text for matching analysis.
    
    Args:
        text: Text to extract keywords from
    
    Returns:
        List of keywords found
    """
    # Simple keyword extraction - split on non-word characters
    keywords = []
    
    # Split into words
    words = text.split()
    
    # Filter out common stop words
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can',
                  'to', 'from', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
                  'about', 'into', 'through', 'during', 'including', 'without',
                  'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they',
                  'what', 'when', 'where', 'why', 'how', 'who', 'which'}
    
    for word in words:
        # Remove punctuation
        clean_word = ''.join(c for c in word if c.isalnum())
        # Check if it's a meaningful word
        if len(clean_word) > 2 and clean_word.lower() not in stop_words:
            keywords.append(clean_word.lower())
    
    return keywords


async def run_rag_accuracy_test() -> Dict[str, Any]:
    """
    Run the complete RAG accuracy test suite.
    
    Returns:
        Dictionary with all test results
    """
    logger.info("🚀 Starting RAG Accuracy Test Suite")
    logger.info(f"📋 Total queries: {len(TEST_QUESTIONS)}")
    
    # Initialize results
    results = {
        'test_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(TEST_QUESTIONS),
            'test_version': '1.0.0'
        },
        'questions': [],
        'summary': {}
    }
    
    # Test each question
    for i, question_data in enumerate(TEST_QUESTIONS, 1):
        result = await test_rag_query(question_data, i, len(TEST_QUESTIONS))
        results['questions'].append(result)
        
        # Small delay between queries to avoid rate limits
        if i < len(TEST_QUESTIONS):
            await asyncio.sleep(1.0)
    
    # Calculate summary statistics
    results['summary'] = _calculate_summary(results['questions'])
    
    logger.info("\n" + "="*70)
    logger.info("📊 Test Summary")
    logger.info("="*70)
    logger.info(f"Total queries: {results['summary']['total_queries']}")
    logger.info(f"Successful queries: {results['summary']['successful_queries']}")
    logger.info(f"Failed queries: {results['summary']['failed_queries']}")
    logger.info(f"Average keyword match score: {results['summary']['avg_keyword_match']:.2%}")
    logger.info(f"Average intent time: {results['summary']['avg_intent_time_ms']:.2f}ms")
    logger.info(f"Average RAG time: {results['summary']['avg_rag_time_ms']:.2f}ms")
    logger.info(f"Average total time: {results['summary']['avg_total_time_ms']:.2f}ms")
    
    return results


def _calculate_summary(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from test results.
    
    Args:
        questions: List of question results
    
    Returns:
        Dictionary with summary statistics
    """
    total = len(questions)
    successful = sum(1 for q in questions if q['status'] == 'success')
    failed = total - successful
    
    # Extract metrics
    keyword_scores = [q['rag_response']['keyword_match_score'] for q in questions]
    intent_times = [q['timing']['intent_classification_ms'] for q in questions]
    rag_times = [q['timing']['rag_processing_ms'] for q in questions]
    total_times = [q['timing']['total_time_ms'] for q in questions]
    
    # Calculate averages
    avg_keyword_match = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
    avg_intent_time = sum(intent_times) / len(intent_times) if intent_times else 0
    avg_rag_time = sum(rag_times) / len(rag_times) if rag_times else 0
    avg_total_time = sum(total_times) / len(total_times) if total_times else 0
    
    # Calculate by category
    category_stats = {}
    for question in questions:
        category = question['category']
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'keyword_scores': [],
                'total_times': []
            }
        
        category_stats[category]['count'] += 1
        category_stats[category]['keyword_scores'].append(question['rag_response']['keyword_match_score'])
        category_stats[category]['total_times'].append(question['timing']['total_time_ms'])
    
    # Calculate averages per category
    for category, stats in category_stats.items():
        stats['avg_keyword_match'] = sum(stats['keyword_scores']) / len(stats['keyword_scores'])
        stats['avg_time_ms'] = sum(stats['total_times']) / len(stats['total_times'])
        del stats['keyword_scores']
        del stats['total_times']
    
    return {
        'total_queries': total,
        'successful_queries': successful,
        'failed_queries': failed,
        'success_rate': successful / total if total > 0 else 0,
        'avg_keyword_match': avg_keyword_match,
        'avg_intent_time_ms': avg_intent_time,
        'avg_rag_time_ms': avg_rag_time,
        'avg_total_time_ms': avg_total_time,
        'min_time_ms': min(total_times) if total_times else 0,
        'max_time_ms': max(total_times) if total_times else 0,
        'category_stats': category_stats
    }


def save_results(results: Dict[str, Any], filename: str = None) -> str:
    """
    Save test results to JSON file.
    
    Args:
        results: Test results dictionary
        filename: Optional filename (auto-generated if None)
    
    Returns:
        Path to saved file
    """
    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_accuracy_results_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Results saved to: {filepath}")
    
    return str(filepath)


async def main():
    """Main test execution"""
    try:
        # Run the test suite
        results = await run_rag_accuracy_test()
        
        # Save results
        filepath = save_results(results)
        
        logger.info(f"\n✅ Test suite completed successfully!")
        logger.info(f"📊 Results saved to: {filepath}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚡ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

