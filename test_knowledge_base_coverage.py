#!/usr/bin/env python3
"""
Knowledge Base Coverage Test Suite for Leibniz University Customer Service Agent

This script validates knowledge base retrieval across all 12 categories, ensuring the RAG
system can retrieve relevant information from all 63 documents and that the vector store
provides comprehensive coverage of university information.

Author: SINDH Technologies
Date: October 2025
"""

import asyncio
import time
import json
import logging
import sys
import os
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Leibniz components
from leibniz_agent.leibniz_rag import get_leibniz_rag, process_leibniz_query_async
from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
KNOWLEDGE_BASE_PATH = "leibniz_knowledge_base"
EXPECTED_CATEGORIES = 12
EXPECTED_TOTAL_DOCUMENTS = 63
RETRIEVAL_QUALITY_THRESHOLD = 0.7
CATEGORY_COVERAGE_TARGET = 1.0  # 100% category coverage

# Category mapping
CATEGORY_MAPPING = {
    "01_university_overview": {"name": "University Overview", "expected_docs": 3},
    "02_faculties_departments": {"name": "Faculties and Departments", "expected_docs": 12},
    "03_admission_enrollment": {"name": "Admission and Enrollment", "expected_docs": 6},
    "04_academic_programs": {"name": "Academic Programs", "expected_docs": 5},
    "05_student_services": {"name": "Student Services", "expected_docs": 7},
    "06_campus_facilities": {"name": "Campus Facilities", "expected_docs": 5},
    "07_academic_policies": {"name": "Academic Policies", "expected_docs": 5},
    "08_administrative_procedures": {"name": "Administrative Procedures", "expected_docs": 4},
    "09_research_opportunities": {"name": "Research Opportunities", "expected_docs": 4},
    "10_campus_life": {"name": "Campus Life", "expected_docs": 4},
    "11_location_transportation": {"name": "Location and Transportation", "expected_docs": 4},
    "12_contact_information": {"name": "Contact Information", "expected_docs": 4}
}

class KnowledgeBaseCoverageTest:
    """Test case for knowledge base category coverage"""
    def __init__(self, category: str, queries: List[str], expected_files: List[str] = None):
        self.category = category
        self.queries = queries
        self.expected_files = expected_files or []

def create_category_test_cases() -> List[KnowledgeBaseCoverageTest]:
    """Create test cases for all 12 knowledge base categories"""
    test_cases = []
    
    # Category 01: University Overview
    test_cases.append(KnowledgeBaseCoverageTest(
        category="01_university_overview",
        queries=[
            "What is Leibniz University?",
            "Tell me about the university's history",
            "What are the university rankings?"
        ],
        expected_files=["general_information.md", "history_mission.md", "rankings_accreditation.md"]
    ))
    
    # Category 02: Faculties and Departments
    test_cases.append(KnowledgeBaseCoverageTest(
        category="02_faculties_departments",
        queries=[
            "Tell me about the computer science department",
            "What programs does the engineering faculty offer?",
            "Information about the law faculty"
        ],
        expected_files=["faculty_electrical_cs.md", "faculty_mechanical.md", "faculty_law.md"]
    ))
    
    # Category 03: Admission and Enrollment
    test_cases.append(KnowledgeBaseCoverageTest(
        category="03_admission_enrollment",
        queries=[
            "What are the bachelor's admission requirements?",
            "How do I apply for a master's program?",
            "What are the application deadlines?"
        ],
        expected_files=["bachelors_admission.md", "masters_admission.md", "deadlines_dates.md"]
    ))
    
    # Category 04: Academic Programs
    test_cases.append(KnowledgeBaseCoverageTest(
        category="04_academic_programs",
        queries=[
            "What bachelor's programs are available?",
            "Tell me about master's programs",
            "What exchange programs does the university offer?"
        ],
        expected_files=["bachelors_programs.md", "masters_programs.md", "exchange_programs.md"]
    ))
    
    # Category 05: Student Services
    test_cases.append(KnowledgeBaseCoverageTest(
        category="05_student_services",
        queries=[
            "How can I get academic advising?",
            "Tell me about campus housing",
            "What financial aid is available?",
            "I need counseling services"
        ],
        expected_files=["academic_advising.md", "student_housing.md", "financial_aid_scholarships.md", "counseling_mental_health.md"]
    ))
    
    # Category 06: Campus Facilities
    test_cases.append(KnowledgeBaseCoverageTest(
        category="06_campus_facilities",
        queries=[
            "Where is the library?",
            "Tell me about study spaces",
            "What dining options are available?",
            "What sports facilities does the campus have?"
        ],
        expected_files=["library_resources.md", "study_spaces.md", "cafeteria_dining.md", "sports_facilities.md"]
    ))
    
    # Category 07: Academic Policies
    test_cases.append(KnowledgeBaseCoverageTest(
        category="07_academic_policies",
        queries=[
            "What are the examination regulations?",
            "How does the grading system work?",
            "What is the academic calendar?"
        ],
        expected_files=["examination_regulations.md", "grading_system.md", "academic_calendar.md"]
    ))
    
    # Category 08: Administrative Procedures
    test_cases.append(KnowledgeBaseCoverageTest(
        category="08_administrative_procedures",
        queries=[
            "How do I register for courses?",
            "How can I request a transcript?",
            "Can I change my program?"
        ],
        expected_files=["registration_procedures.md", "transcript_requests.md", "change_of_program.md"]
    ))
    
    # Category 09: Research Opportunities
    test_cases.append(KnowledgeBaseCoverageTest(
        category="09_research_opportunities",
        queries=[
            "What research centers are available?",
            "Can undergraduates do research?",
            "Tell me about graduate research programs"
        ],
        expected_files=["research_centers.md", "undergraduate_research.md", "graduate_research.md"]
    ))
    
    # Category 10: Campus Life
    test_cases.append(KnowledgeBaseCoverageTest(
        category="10_campus_life",
        queries=[
            "What student organizations are there?",
            "What events happen on campus?",
            "Are there sports clubs?"
        ],
        expected_files=["student_organizations.md", "events_activities.md", "sports_clubs.md"]
    ))
    
    # Category 11: Location and Transportation
    test_cases.append(KnowledgeBaseCoverageTest(
        category="11_location_transportation",
        queries=[
            "How do I get to campus?",
            "Where can I park?",
            "Tell me about living in Hannover"
        ],
        expected_files=["public_transportation.md", "parking_information.md", "hannover_city_guide.md"]
    ))
    
    # Category 12: Contact Information
    test_cases.append(KnowledgeBaseCoverageTest(
        category="12_contact_information",
        queries=[
            "How can I contact the admissions office?",
            "What are the emergency contacts?",
            "What are the office hours?"
        ],
        expected_files=["department_contacts.md", "emergency_contacts.md", "office_hours.md"]
    ))
    
    return test_cases

async def verify_knowledge_base_structure() -> Dict[str, Any]:
    """Verify knowledge base structure and document inventory"""
    logger.info("Verifying knowledge base structure...")
    
    results = {
        "structure_check": {},
        "document_inventory": {},
        "summary": {
            "total_categories": 0,
            "total_documents": 0,
            "missing_categories": [],
            "missing_documents": [],
            "structure_valid": False
        },
        "issues": []
    }
    
    try:
        # Check if knowledge base directory exists
        kb_path = Path(KNOWLEDGE_BASE_PATH)
        if not kb_path.exists():
            results["issues"].append(f"Knowledge base directory not found: {KNOWLEDGE_BASE_PATH}")
            return results
        
        logger.info(f"Knowledge base found: {KNOWLEDGE_BASE_PATH}")
        
        # Check each category directory
        found_categories = 0
        total_documents = 0
        
        for category_dir, info in CATEGORY_MAPPING.items():
            category_path = kb_path / category_dir
            
            if category_path.exists():
                found_categories += 1
                
                # Count documents in category
                md_files = list(category_path.glob("*.md"))
                # Exclude README files
                md_files = [f for f in md_files if not f.name.startswith("00-")]
                
                doc_count = len(md_files)
                total_documents += doc_count
                
                results["document_inventory"][category_dir] = {
                    "expected": info["expected_docs"],
                    "found": doc_count,
                    "files": [f.name for f in md_files],
                    "complete": doc_count >= info["expected_docs"]
                }
                
                logger.info(f"Category {category_dir}: {doc_count}/{info['expected_docs']} documents")
                
                if doc_count < info["expected_docs"]:
                    results["issues"].append(f"Category {category_dir} missing documents: {doc_count}/{info['expected_docs']}")
            else:
                results["summary"]["missing_categories"].append(category_dir)
                results["issues"].append(f"Category directory missing: {category_dir}")
        
        results["summary"]["total_categories"] = found_categories
        results["summary"]["total_documents"] = total_documents
        results["summary"]["structure_valid"] = (found_categories == EXPECTED_CATEGORIES and 
                                               total_documents >= EXPECTED_TOTAL_DOCUMENTS)
        
        logger.info(f"Categories found: {found_categories}/{EXPECTED_CATEGORIES}")
        logger.info(f"Total documents: {total_documents} (expected: ≥{EXPECTED_TOTAL_DOCUMENTS})")
        logger.info(f"Structure valid: {'Yes ✅' if results['summary']['structure_valid'] else 'No ❌'}")
        
    except Exception as e:
        logger.error(f"Error verifying knowledge base structure: {str(e)}")
        results["issues"].append(f"Structure verification error: {str(e)}")
    
    return results

async def test_vector_store_coverage() -> Dict[str, Any]:
    """Test vector store coverage and completeness"""
    logger.info("Testing vector store coverage...")
    
    results = {
        "vector_store_stats": {},
        "summary": {
            "vector_store_exists": False,
            "total_chunks": 0,
            "categories_represented": 0,
            "coverage_complete": False
        },
        "issues": []
    }
    
    try:
        # Initialize RAG system
        rag = get_leibniz_rag()
        
        # Get vector store statistics
        stats = rag.get_vector_store_stats()
        
        results["vector_store_stats"] = stats
        results["summary"]["vector_store_exists"] = stats["vector_store_size"] > 0
        results["summary"]["total_chunks"] = stats["total_documents"]
        results["summary"]["categories_represented"] = stats["categories_loaded"]
        
        # Validate coverage
        expected_min_chunks = EXPECTED_TOTAL_DOCUMENTS * 2  # Assume ~2 chunks per document
        coverage_adequate = (stats["total_documents"] >= expected_min_chunks and 
                           stats["categories_loaded"] >= EXPECTED_CATEGORIES)
        
        results["summary"]["coverage_complete"] = coverage_adequate
        
        logger.info(f"Vector store size: {stats['vector_store_size']}")
        logger.info(f"Total chunks: {stats['total_documents']}")
        logger.info(f"Categories loaded: {stats['categories_loaded']}/{EXPECTED_CATEGORIES}")
        logger.info(f"Coverage complete: {'Yes ✅' if coverage_adequate else 'No ❌'}")
        
        if not coverage_adequate:
            if stats["total_documents"] < expected_min_chunks:
                results["issues"].append(f"Vector store has only {stats['total_documents']} chunks (expected: ≥{expected_min_chunks})")
            if stats["categories_loaded"] < EXPECTED_CATEGORIES:
                results["issues"].append(f"Only {stats['categories_loaded']} categories loaded (expected: {EXPECTED_CATEGORIES})")
        
    except Exception as e:
        logger.error(f"Error testing vector store coverage: {str(e)}")
        results["issues"].append(f"Vector store coverage error: {str(e)}")
    
    return results

async def test_category_retrieval(test_cases: List[KnowledgeBaseCoverageTest]) -> Dict[str, Any]:
    """Test retrieval from all 12 knowledge base categories"""
    logger.info("Testing category-specific retrieval...")
    
    results = {
        "category_tests": [],
        "summary": {
            "total_categories": len(test_cases),
            "categories_passed": 0,
            "avg_retrieval_quality": 0.0,
            "all_categories_covered": False
        },
        "issues": []
    }
    
    try:
        # Initialize services
        await get_leibniz_services_manager()
        
        total_quality_score = 0.0
        
        for test_case in test_cases:
            category_name = CATEGORY_MAPPING[test_case.category]["name"]
            logger.info(f"\nTesting category: {category_name}")
            
            category_results = {
                "category": test_case.category,
                "category_name": category_name,
                "query_results": [],
                "avg_quality": 0.0,
                "category_passed": False
            }
            
            query_quality_scores = []
            
            for query in test_case.queries:
                logger.info(f"Query: '{query}'")
                
                start_time = time.time()
                
                try:
                    # Create context for the query
                    context = {
                        "user_goal": f"asking about {category_name.lower()}",
                        "key_entities": {"category": test_case.category},
                        "extracted_meaning": query.lower()
                    }
                    
                    # Process RAG query
                    response = await process_leibniz_query_async(context=context, query=query)
                    
                    processing_time = time.time() - start_time
                    
                    # Validate response quality
                    quality_score = validate_retrieval_quality(query, response, test_case.category)
                    query_quality_scores.append(quality_score)
                    
                    query_result = {
                        "query": query,
                        "response": response[:200] + "..." if len(response) > 200 else response,
                        "response_length": len(response),
                        "processing_time": processing_time,
                        "quality_score": quality_score,
                        "quality_good": quality_score >= RETRIEVAL_QUALITY_THRESHOLD
                    }
                    
                    category_results["query_results"].append(query_result)
                    
                    logger.info(f"Response length: {len(response)} chars")
                    logger.info(f"Quality score: {quality_score:.2f}")
                    logger.info(f"Processing time: {processing_time:.2f}s")
                    logger.info(f"Quality: {'Good ✅' if quality_score >= RETRIEVAL_QUALITY_THRESHOLD else 'Poor ❌'}")
                
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    results["issues"].append(f"Query error in {category_name}: {str(e)}")
                    
                    query_result = {
                        "query": query,
                        "error": str(e),
                        "quality_score": 0.0,
                        "quality_good": False
                    }
                    
                    category_results["query_results"].append(query_result)
                    query_quality_scores.append(0.0)
            
            # Calculate category average quality
            if query_quality_scores:
                category_avg_quality = sum(query_quality_scores) / len(query_quality_scores)
                category_results["avg_quality"] = category_avg_quality
                category_results["category_passed"] = category_avg_quality >= RETRIEVAL_QUALITY_THRESHOLD
                
                if category_results["category_passed"]:
                    results["summary"]["categories_passed"] += 1
                
                total_quality_score += category_avg_quality
                
                logger.info(f"Category {category_name} average quality: {category_avg_quality:.2f}")
                logger.info(f"Category result: {'PASS ✅' if category_results['category_passed'] else 'FAIL ❌'}")
            
            results["category_tests"].append(category_results)
        
        # Calculate overall metrics
        if results["summary"]["total_categories"] > 0:
            results["summary"]["avg_retrieval_quality"] = total_quality_score / results["summary"]["total_categories"]
            results["summary"]["all_categories_covered"] = results["summary"]["categories_passed"] == results["summary"]["total_categories"]
        
        logger.info(f"\nCategory Coverage Summary:")
        logger.info(f"Categories passed: {results['summary']['categories_passed']}/{results['summary']['total_categories']}")
        logger.info(f"Average quality: {results['summary']['avg_retrieval_quality']:.2f}")
        logger.info(f"All categories covered: {'Yes ✅' if results['summary']['all_categories_covered'] else 'No ❌'}")
        
    except Exception as e:
        logger.error(f"Error in category retrieval test: {str(e)}")
        results["issues"].append(f"Category retrieval error: {str(e)}")
    
    return results

def validate_retrieval_quality(query: str, response: str, category: str) -> float:
    """Validate retrieval quality for a query-response pair"""
    quality_score = 1.0
    
    # Check response not empty
    if not response or len(response) < 50:
        quality_score -= 0.4
    
    # Check response addresses query (simple keyword matching)
    query_keywords = [word.lower() for word in query.split() if len(word) > 3]
    response_lower = response.lower()
    
    keyword_matches = sum(1 for keyword in query_keywords if keyword in response_lower)
    keyword_coverage = keyword_matches / len(query_keywords) if query_keywords else 0
    
    if keyword_coverage < 0.3:
        quality_score -= 0.3
    elif keyword_coverage < 0.6:
        quality_score -= 0.1
    
    # Check for category-specific terms
    category_terms = {
        "01_university_overview": ["leibniz", "university", "history", "mission"],
        "02_faculties_departments": ["faculty", "department", "program", "professor"],
        "03_admission_enrollment": ["admission", "application", "requirement", "deadline"],
        "04_academic_programs": ["program", "degree", "bachelor", "master"],
        "05_student_services": ["service", "support", "advising", "housing", "financial"],
        "06_campus_facilities": ["library", "facility", "building", "campus"],
        "07_academic_policies": ["policy", "regulation", "exam", "grade"],
        "08_administrative_procedures": ["procedure", "registration", "transcript"],
        "09_research_opportunities": ["research", "center", "opportunity"],
        "10_campus_life": ["student", "organization", "event", "club"],
        "11_location_transportation": ["location", "transport", "parking", "hannover"],
        "12_contact_information": ["contact", "phone", "email", "office", "hours"]
    }
    
    relevant_terms = category_terms.get(category, [])
    term_matches = sum(1 for term in relevant_terms if term in response_lower)
    term_coverage = term_matches / len(relevant_terms) if relevant_terms else 1.0
    
    if term_coverage < 0.3:
        quality_score -= 0.2
    
    # Check response length (not too short, not too long)
    if len(response) < 100:
        quality_score -= 0.1
    elif len(response) > 800:
        quality_score -= 0.05
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, quality_score))

async def test_cross_category_retrieval() -> Dict[str, Any]:
    """Test queries that should retrieve from multiple categories"""
    logger.info("Testing cross-category retrieval...")
    
    results = {
        "cross_category_tests": [],
        "summary": {
            "total_tests": 0,
            "multi_category_success": 0,
            "avg_categories_per_query": 0.0
        },
        "issues": []
    }
    
    # Multi-category test queries
    multi_category_queries = [
        {
            "query": "How do I apply for the CS program and what financial aid is available?",
            "expected_categories": ["03_admission_enrollment", "04_academic_programs", "05_student_services"],
            "description": "Should cover admission, programs, and financial aid"
        },
        {
            "query": "Tell me about campus housing and transportation options",
            "expected_categories": ["05_student_services", "11_location_transportation"],
            "description": "Should cover housing and transportation"
        },
        {
            "query": "What are the exam policies and when is the academic calendar?",
            "expected_categories": ["07_academic_policies"],
            "description": "Should cover academic policies"
        }
    ]
    
    results["summary"]["total_tests"] = len(multi_category_queries)
    total_categories_retrieved = 0
    
    try:
        for test_query in multi_category_queries:
            logger.info(f"\nTesting multi-category query: '{test_query['query'][:50]}...'")
            
            # Create context
            context = {
                "user_goal": "asking about multiple university topics",
                "key_entities": {"topic": "multiple"},
                "extracted_meaning": test_query["query"].lower()
            }
            
            # Process query
            response = await process_leibniz_query_async(context=context, query=test_query["query"])
            
            # Analyze response for category coverage (simplified)
            categories_covered = []
            response_lower = response.lower()
            
            # Check for category-specific keywords in response
            for category, terms in {
                "03_admission_enrollment": ["admission", "application", "requirement"],
                "04_academic_programs": ["program", "degree", "course"],
                "05_student_services": ["housing", "financial aid", "service"],
                "07_academic_policies": ["exam", "policy", "regulation", "calendar"],
                "11_location_transportation": ["transport", "parking", "location"]
            }.items():
                if any(term in response_lower for term in terms):
                    categories_covered.append(category)
            
            # Check if expected categories are covered
            expected_covered = sum(1 for cat in test_query["expected_categories"] if cat in categories_covered)
            coverage_rate = expected_covered / len(test_query["expected_categories"]) if test_query["expected_categories"] else 0
            
            multi_category_success = coverage_rate >= 0.5  # At least 50% of expected categories
            
            if multi_category_success:
                results["summary"]["multi_category_success"] += 1
            
            total_categories_retrieved += len(categories_covered)
            
            test_result = {
                "query": test_query["query"],
                "expected_categories": test_query["expected_categories"],
                "categories_covered": categories_covered,
                "coverage_rate": coverage_rate,
                "response_length": len(response),
                "success": multi_category_success
            }
            
            results["cross_category_tests"].append(test_result)
            
            logger.info(f"Expected categories: {len(test_query['expected_categories'])}")
            logger.info(f"Categories covered: {len(categories_covered)}")
            logger.info(f"Coverage rate: {coverage_rate:.1%}")
            logger.info(f"Success: {'Yes ✅' if multi_category_success else 'No ❌'}")
        
        # Calculate average categories per query
        if results["summary"]["total_tests"] > 0:
            results["summary"]["avg_categories_per_query"] = total_categories_retrieved / results["summary"]["total_tests"]
        
        logger.info(f"\nCross-category Summary:")
        logger.info(f"Multi-category success: {results['summary']['multi_category_success']}/{results['summary']['total_tests']}")
        logger.info(f"Avg categories per query: {results['summary']['avg_categories_per_query']:.1f}")
        
    except Exception as e:
        logger.error(f"Error in cross-category test: {str(e)}")
        results["issues"].append(f"Cross-category error: {str(e)}")
    
    return results

def generate_knowledge_base_coverage_report(structure_results: Dict, vector_results: Dict, 
                                          category_results: Dict, cross_category_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive knowledge base coverage report"""
    report = {
        "test_suite": "Knowledge Base Coverage Tests",
        "execution_time": datetime.now().isoformat(),
        "structure_verification": structure_results,
        "vector_store_coverage": vector_results,
        "category_retrieval": category_results,
        "cross_category_retrieval": cross_category_results,
        "summary": {
            "knowledge_base_complete": structure_results["summary"]["structure_valid"],
            "vector_store_ready": vector_results["summary"]["vector_store_exists"],
            "category_coverage": category_results["summary"]["categories_passed"] / category_results["summary"]["total_categories"] if category_results["summary"]["total_categories"] > 0 else 0,
            "retrieval_quality": category_results["summary"]["avg_retrieval_quality"],
            "multi_category_success": cross_category_results["summary"]["multi_category_success"] / cross_category_results["summary"]["total_tests"] if cross_category_results["summary"]["total_tests"] > 0 else 0,
            "overall_coverage": "excellent"
        },
        "issues": [],
        "recommendations": []
    }
    
    # Collect all issues
    for result_set in [structure_results, vector_results, category_results, cross_category_results]:
        report["issues"].extend(result_set.get("issues", []))
    
    # Determine overall coverage quality
    coverage_score = (
        (1.0 if report["summary"]["knowledge_base_complete"] else 0.0) +
        (1.0 if report["summary"]["vector_store_ready"] else 0.0) +
        (report["summary"]["category_coverage"]) +
        (report["summary"]["retrieval_quality"]) +
        (report["summary"]["multi_category_success"])
    ) / 5.0
    
    if coverage_score >= 0.9:
        report["summary"]["overall_coverage"] = "excellent"
    elif coverage_score >= 0.7:
        report["summary"]["overall_coverage"] = "good"
    else:
        report["summary"]["overall_coverage"] = "needs_improvement"
    
    # Generate recommendations
    if not report["summary"]["knowledge_base_complete"]:
        report["recommendations"].append({
            "priority": "critical",
            "category": "knowledge_base",
            "recommendation": "Fix knowledge base structure - missing categories or documents"
        })
    
    if not report["summary"]["vector_store_ready"]:
        report["recommendations"].append({
            "priority": "critical",
            "category": "vector_store",
            "recommendation": "Fix vector store - not properly loaded or empty"
        })
    
    if report["summary"]["category_coverage"] < 0.8:
        report["recommendations"].append({
            "priority": "high",
            "category": "retrieval",
            "recommendation": f"Improve category retrieval - only {report['summary']['category_coverage']:.1%} categories passing"
        })
    
    if report["summary"]["retrieval_quality"] < RETRIEVAL_QUALITY_THRESHOLD:
        report["recommendations"].append({
            "priority": "high",
            "category": "quality",
            "recommendation": f"Improve retrieval quality - average {report['summary']['retrieval_quality']:.2f} below threshold {RETRIEVAL_QUALITY_THRESHOLD}"
        })
    
    if report["summary"]["multi_category_success"] < 0.7:
        report["recommendations"].append({
            "priority": "medium",
            "category": "cross_category",
            "recommendation": f"Improve cross-category retrieval - only {report['summary']['multi_category_success']:.1%} success rate"
        })
    
    return report

def print_report_summary(report: Dict):
    """Print formatted report summary to console"""
    print("\n" + "="*80)
    print("KNOWLEDGE BASE COVERAGE TEST RESULTS")
    print("="*80)
    
    print(f"\nOverall Coverage: {report['summary']['overall_coverage'].upper()}")
    print(f"Knowledge Base Complete: {'Yes ✅' if report['summary']['knowledge_base_complete'] else 'No ❌'}")
    print(f"Vector Store Ready: {'Yes ✅' if report['summary']['vector_store_ready'] else 'No ❌'}")
    
    print(f"\nCategory Coverage:")
    print(f"  Categories Passing: {report['summary']['category_coverage']:.1%}")
    print(f"  Average Quality: {report['summary']['retrieval_quality']:.2f}")
    
    print(f"\nVector Store:")
    vs_stats = report["vector_store_coverage"]["vector_store_stats"]
    print(f"  Total Chunks: {vs_stats.get('total_documents', 0)}")
    print(f"  Categories Loaded: {vs_stats.get('categories_loaded', 0)}/{EXPECTED_CATEGORIES}")
    
    print(f"\nCross-Category:")
    print(f"  Multi-category Success: {report['summary']['multi_category_success']:.1%}")
    
    if report["issues"]:
        print(f"\nIssues Found ({len(report['issues'])}):")
        for issue in report["issues"][:5]:
            print(f"  - {issue}")
        if len(report["issues"]) > 5:
            print(f"  ... and {len(report['issues']) - 5} more issues")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - [{rec['priority'].upper()}] {rec['recommendation']}")
    
    print("\n" + "="*80)

async def main():
    """Main test runner"""
    logger.info("Starting Leibniz Knowledge Base Coverage Tests")
    
    # Verify knowledge base structure
    structure_results = await verify_knowledge_base_structure()
    
    # Test vector store coverage
    vector_results = await test_vector_store_coverage()
    
    # Create category test cases
    test_cases = create_category_test_cases()
    
    # Test category-specific retrieval
    category_results = await test_category_retrieval(test_cases)
    
    # Test cross-category retrieval
    cross_category_results = await test_cross_category_retrieval()
    
    # Generate report
    report = generate_knowledge_base_coverage_report(
        structure_results, vector_results, category_results, cross_category_results
    )
    
    # Save results
    output_dir = Path("leibniz_agent/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "knowledge_base_coverage_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save markdown report
    with open(output_dir / "KNOWLEDGE_BASE_COVERAGE_REPORT.md", "w") as f:
        f.write("# Knowledge Base Coverage Test Report\n\n")
        f.write(f"Generated: {report['execution_time']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Overall Coverage: {report['summary']['overall_coverage'].upper()}\n")
        f.write(f"- Knowledge Base Complete: {'Yes' if report['summary']['knowledge_base_complete'] else 'No'}\n")
        f.write(f"- Category Coverage: {report['summary']['category_coverage']:.1%}\n")
        f.write(f"- Average Retrieval Quality: {report['summary']['retrieval_quality']:.2f}\n")
        f.write(f"- Multi-category Success: {report['summary']['multi_category_success']:.1%}\n\n")
        
        f.write("## Knowledge Base Structure\n\n")
        f.write(f"- Total Categories: {report['structure_verification']['summary']['total_categories']}/{EXPECTED_CATEGORIES}\n")
        f.write(f"- Total Documents: {report['structure_verification']['summary']['total_documents']}\n")
        f.write(f"- Structure Valid: {'Yes' if report['structure_verification']['summary']['structure_valid'] else 'No'}\n\n")
        
        f.write("## Vector Store Coverage\n\n")
        vs_stats = report["vector_store_coverage"]["vector_store_stats"]
        f.write(f"- Vector Store Size: {vs_stats.get('vector_store_size', 0)}\n")
        f.write(f"- Total Chunks: {vs_stats.get('total_documents', 0)}\n")
        f.write(f"- Categories Loaded: {vs_stats.get('categories_loaded', 0)}\n\n")
        
        f.write("## Category Retrieval Results\n\n")
        for test in report['category_retrieval']['category_tests']:
            f.write(f"### {test['category_name']}\n")
            f.write(f"- Average Quality: {test['avg_quality']:.2f}\n")
            f.write(f"- Result: {'PASS' if test['category_passed'] else 'FAIL'}\n")
            f.write(f"- Queries Tested: {len(test['query_results'])}\n\n")
        
        if report['issues']:
            f.write("## Issues Found\n\n")
            for issue in report['issues']:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        if report['recommendations']:
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']}\n")
    
    # Print summary
    print_report_summary(report)
    
    # Return exit code
    return 0 if report["summary"]["overall_coverage"] in ["excellent", "good"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
