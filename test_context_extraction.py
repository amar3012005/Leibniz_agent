#!/usr/bin/env python3
"""
Context Extraction Test Suite for Leibniz University Customer Service Agent

This script validates context extraction in the intent parser and context-based RAG retrieval.
It tests that the intent parser correctly extracts structured context (user_goal, key_entities, 
extracted_meaning) and that the RAG system uses this context for better retrieval.

Author: SINDH Technologies
Date: October 2025
"""

import asyncio
import time
import json
import logging
import sys
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Leibniz components
from leibniz_agent.leibniz_intent_parser import classify_leibniz_intent
from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager, process_leibniz_rag_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration constants
CONTEXT_QUALITY_THRESHOLD = 0.7  # Minimum context quality score
FAST_ROUTE_TARGET = 0.8  # Target percentage for fast route usage
CONTEXT_IMPROVEMENT_THRESHOLD = 0.1  # Minimum improvement from context-aware RAG

class ContextExtractionTestCase:
    """Defines a test case for context extraction validation"""
    def __init__(self, name: str, input_text: str, expected_intent: str, 
                 expected_context: Dict[str, Any], description: str = ""):
        self.name = name
        self.input_text = input_text
        self.expected_intent = expected_intent
        self.expected_context = expected_context
        self.description = description

def create_context_test_cases() -> List[ContextExtractionTestCase]:
    """Create comprehensive test cases for context extraction"""
    test_cases = []
    
    # Test Case 1: Simple RAG Query
    test_cases.append(ContextExtractionTestCase(
        name="Simple RAG Query",
        input_text="What are the requirements for the computer science program?",
        expected_intent="RAG_QUERY",
        expected_context={
            "user_goal": "asking about computer science program admission requirements",
            "key_entities": {"program": "computer science", "topic": "requirements"},
            "extracted_meaning": "computer science program admission requirements"
        },
        description="Tests basic context extraction for academic program inquiry"
    ))
    
    # Test Case 2: Appointment Request
    test_cases.append(ContextExtractionTestCase(
        name="Appointment Request",
        input_text="I'd like to schedule an appointment with the admissions office for next Tuesday",
        expected_intent="APPOINTMENT_SCHEDULING",
        expected_context={
            "user_goal": "wants to schedule appointment with admissions office",
            "key_entities": {"department": "admissions", "datetime": "next tuesday"},
            "extracted_meaning": "schedule appointment admissions office"
        },
        description="Tests context extraction for appointment scheduling"
    ))
    
    # Test Case 3: Complex Multi-Entity Query
    test_cases.append(ContextExtractionTestCase(
        name="Complex Multi-Entity Query",
        input_text="Can you tell me about financial aid scholarships for international students in the engineering program?",
        expected_intent="RAG_QUERY",
        expected_context={
            "user_goal": "asking about financial aid scholarships for international students",
            "key_entities": {
                "service": "financial aid", 
                "topic": "scholarships", 
                "student_type": "international", 
                "program": "engineering"
            },
            "extracted_meaning": "financial aid scholarships international students engineering program"
        },
        description="Tests complex multi-entity extraction"
    ))
    
    # Test Case 4: Vague Query (Edge Case)
    test_cases.append(ContextExtractionTestCase(
        name="Vague Query",
        input_text="Tell me more",
        expected_intent="UNCLEAR",
        expected_context={
            "user_goal": "asking for more information",
            "key_entities": {},
            "extracted_meaning": "more information"
        },
        description="Tests handling of vague input with minimal context"
    ))
    
    # Test Case 5: Greeting (No Context Needed)
    test_cases.append(ContextExtractionTestCase(
        name="Greeting",
        input_text="Hello",
        expected_intent="GREETING",
        expected_context={
            "user_goal": "greeting the assistant",
            "key_entities": {},
            "extracted_meaning": ""
        },
        description="Tests minimal context for non-informational intents"
    ))
    
    # Test Case 6: Housing Query
    test_cases.append(ContextExtractionTestCase(
        name="Housing Query",
        input_text="What housing options are available for graduate students?",
        expected_intent="RAG_QUERY",
        expected_context={
            "user_goal": "asking about housing options for graduate students",
            "key_entities": {"service": "housing", "student_type": "graduate"},
            "extracted_meaning": "housing options graduate students"
        },
        description="Tests service-specific context extraction"
    ))
    
    # Test Case 7: Library Query
    test_cases.append(ContextExtractionTestCase(
        name="Library Query",
        input_text="Where is the library and what are the opening hours?",
        expected_intent="RAG_QUERY",
        expected_context={
            "user_goal": "asking about library location and hours",
            "key_entities": {"facility": "library", "topic": "location hours"},
            "extracted_meaning": "library location opening hours"
        },
        description="Tests facility-related context extraction"
    ))
    
    # Test Case 8: Exit Intent
    test_cases.append(ContextExtractionTestCase(
        name="Exit Intent",
        input_text="Thanks, goodbye",
        expected_intent="EXIT",
        expected_context={
            "user_goal": "ending the conversation",
            "key_entities": {},
            "extracted_meaning": ""
        },
        description="Tests minimal context for conversation ending"
    ))
    
    return test_cases

def validate_context_quality(context: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """Validate extracted context quality"""
    issues = []
    scores = {}
    
    # Validate user_goal
    user_goal = context.get("user_goal", "")
    if not user_goal and input_text.lower() not in ["hello", "hi", "bye", "goodbye", "thanks"]:
        issues.append("Missing user_goal for informational query")
        scores["user_goal"] = 0.0
    elif user_goal:
        # Check for action verbs
        action_verbs = ["asking", "wants", "inquiring", "requesting", "seeking", "looking"]
        has_action = any(verb in user_goal.lower() for verb in action_verbs)
        
        # Check length
        length_ok = 5 <= len(user_goal) <= 100
        
        # Check relevance to input
        input_words = set(input_text.lower().split())
        goal_words = set(user_goal.lower().split())
        overlap = len(input_words.intersection(goal_words)) / len(input_words) if input_words else 0
        
        goal_score = (0.4 if has_action else 0.0) + (0.3 if length_ok else 0.0) + (0.3 * overlap)
        scores["user_goal"] = min(goal_score, 1.0)
        
        if goal_score < 0.5:
            issues.append(f"Low quality user_goal: {user_goal}")
    else:
        scores["user_goal"] = 1.0  # OK for greetings/exits
    
    # Validate key_entities
    key_entities = context.get("key_entities", {})
    if isinstance(key_entities, dict):
        entity_score = 1.0
        
        # Check entity relevance
        input_lower = input_text.lower()
        for entity_type, entity_value in key_entities.items():
            if isinstance(entity_value, str):
                if entity_value.lower() not in input_lower:
                    entity_score -= 0.2
                    issues.append(f"Entity '{entity_value}' not found in input")
        
        scores["key_entities"] = max(entity_score, 0.0)
    else:
        issues.append("key_entities should be a dictionary")
        scores["key_entities"] = 0.0
    
    # Validate extracted_meaning
    extracted_meaning = context.get("extracted_meaning", "")
    if extracted_meaning:
        # Check for filler words removal
        filler_words = ["um", "uh", "like", "you know", "actually", "basically"]
        has_fillers = any(filler in extracted_meaning.lower() for filler in filler_words)
        
        # Check for question word removal
        question_words = ["what", "how", "where", "when", "why", "can you", "could you"]
        has_questions = any(qw in extracted_meaning.lower() for qw in question_words)
        
        # Check length
        length_ok = 3 <= len(extracted_meaning) <= 200
        
        # Check normalization (lowercase, single spaces)
        is_normalized = extracted_meaning.islower() and "  " not in extracted_meaning
        
        meaning_score = (0.3 if not has_fillers else 0.0) + \
                       (0.2 if not has_questions else 0.0) + \
                       (0.3 if length_ok else 0.0) + \
                       (0.2 if is_normalized else 0.0)
        
        scores["extracted_meaning"] = meaning_score
        
        if has_fillers:
            issues.append("Filler words not removed from extracted_meaning")
        if not is_normalized:
            issues.append("extracted_meaning not properly normalized")
    else:
        # Empty is OK for greetings/exits
        if input_text.lower() in ["hello", "hi", "bye", "goodbye", "thanks"]:
            scores["extracted_meaning"] = 1.0
        else:
            scores["extracted_meaning"] = 0.0
            issues.append("Missing extracted_meaning for informational query")
    
    # Calculate overall quality score
    overall_score = sum(scores.values()) / len(scores) if scores else 0.0
    
    return {
        "valid": overall_score >= CONTEXT_QUALITY_THRESHOLD,
        "overall_score": overall_score,
        "component_scores": scores,
        "issues": issues,
        "quality_level": "high" if overall_score >= 0.8 else "medium" if overall_score >= 0.6 else "low"
    }

async def test_context_extraction(test_cases: List[ContextExtractionTestCase]) -> Dict[str, Any]:
    """Test context extraction for all test cases"""
    logger.info("Testing context extraction...")
    
    results = {
        "test_cases": [],
        "summary": {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "context_quality_scores": [],
            "fast_route_count": 0,
            "gemini_route_count": 0
        },
        "issues": []
    }
    
    # Initialize services
    await get_leibniz_services_manager()
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case.name}")
        logger.info(f"Input: '{test_case.input_text}'")
        
        start_time = time.time()
        
        try:
            # Classify intent and extract context
            result = await classify_leibniz_intent(test_case.input_text)
            
            processing_time = time.time() - start_time
            
            # Extract results
            actual_intent = result.get("intent", "UNKNOWN")
            actual_context = result.get("entities", {})
            fast_route = result.get("fast_route", False)
            confidence = result.get("confidence", 0.0)
            
            # Track route usage
            if fast_route:
                results["summary"]["fast_route_count"] += 1
            else:
                results["summary"]["gemini_route_count"] += 1
            
            # Validate intent
            intent_correct = actual_intent == test_case.expected_intent
            
            # Validate context quality
            context_validation = validate_context_quality(actual_context, test_case.input_text)
            results["summary"]["context_quality_scores"].append(context_validation["overall_score"])
            
            # Check if test passed
            test_passed = intent_correct and context_validation["valid"]
            
            if test_passed:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
            
            # Store test result
            test_result = {
                "name": test_case.name,
                "input": test_case.input_text,
                "expected_intent": test_case.expected_intent,
                "actual_intent": actual_intent,
                "intent_correct": intent_correct,
                "expected_context": test_case.expected_context,
                "actual_context": actual_context,
                "context_validation": context_validation,
                "fast_route": fast_route,
                "confidence": confidence,
                "processing_time": processing_time,
                "passed": test_passed
            }
            
            results["test_cases"].append(test_result)
            
            logger.info(f"Intent: {actual_intent} ({'' if intent_correct else ''})")
            logger.info(f"Context quality: {context_validation['overall_score']:.2f} ({context_validation['quality_level']})")
            logger.info(f"Fast route: {'Yes' if fast_route else 'No'}")
            logger.info(f"Result: {'PASS' if test_passed else 'FAIL'}")
            
            if not test_passed:
                issues = []
                if not intent_correct:
                    issues.append(f"Intent mismatch: expected {test_case.expected_intent}, got {actual_intent}")
                if not context_validation["valid"]:
                    issues.extend(context_validation["issues"])
                
                results["issues"].extend(issues)
                logger.warning(f"Issues: {'; '.join(issues)}")
        
        except Exception as e:
            logger.error(f"Error testing {test_case.name}: {str(e)}")
            results["summary"]["failed"] += 1
            results["issues"].append(f"Exception in {test_case.name}: {str(e)}")
            
            results["test_cases"].append({
                "name": test_case.name,
                "input": test_case.input_text,
                "error": str(e),
                "passed": False
            })
    
    return results

async def test_context_aware_rag_comparison() -> Dict[str, Any]:
    """Test context-aware RAG vs raw query comparison"""
    logger.info("\nTesting context-aware RAG improvement...")
    
    test_queries = [
        {
            "query": "What are CS program requirements?",
            "context": {
                "user_goal": "asking about computer science program admission requirements",
                "key_entities": {"program": "computer science", "topic": "requirements"},
                "extracted_meaning": "computer science program admission requirements prerequisites"
            }
        },
        {
            "query": "Tell me about housing",
            "context": {
                "user_goal": "asking about student housing options",
                "key_entities": {"service": "housing", "topic": "options"},
                "extracted_meaning": "student housing options"
            }
        },
        {
            "query": "Financial aid info",
            "context": {
                "user_goal": "asking about financial aid information",
                "key_entities": {"service": "financial aid", "topic": "information"},
                "extracted_meaning": "financial aid information"
            }
        }
    ]
    
    results = {
        "comparisons": [],
        "summary": {
            "total_tests": len(test_queries),
            "context_improved": 0,
            "no_improvement": 0,
            "context_worse": 0,
            "avg_improvement": 0.0
        }
    }
    
    for i, test_query in enumerate(test_queries):
        logger.info(f"\nTesting query {i+1}: '{test_query['query']}'")
        
        try:
            # Test with context
            start_time = time.time()
            context_result = await process_leibniz_rag_query(
                text=test_query["query"],
                context=test_query["context"]
            )
            context_time = time.time() - start_time
            
            # Test without context (raw query)
            start_time = time.time()
            raw_result = await process_leibniz_rag_query(
                text=test_query["query"],
                context=None
            )
            raw_time = time.time() - start_time
            
            # Compare results (simplified comparison based on response length and content)
            context_response = str(context_result)
            raw_response = str(raw_result)
            
            # Simple quality metrics
            context_length = len(context_response)
            raw_length = len(raw_response)
            
            # Check if context response mentions key entities
            key_entities = test_query["context"].get("key_entities", {})
            context_entity_mentions = sum(1 for entity in key_entities.values() 
                                        if str(entity).lower() in context_response.lower())
            raw_entity_mentions = sum(1 for entity in key_entities.values() 
                                    if str(entity).lower() in raw_response.lower())
            
            # Calculate improvement score
            length_improvement = (context_length - raw_length) / max(raw_length, 1)
            entity_improvement = context_entity_mentions - raw_entity_mentions
            
            # Overall improvement score
            improvement_score = (length_improvement * 0.3) + (entity_improvement * 0.7)
            
            # Categorize improvement
            if improvement_score > CONTEXT_IMPROVEMENT_THRESHOLD:
                results["summary"]["context_improved"] += 1
                improvement_category = "improved"
            elif improvement_score < -CONTEXT_IMPROVEMENT_THRESHOLD:
                results["summary"]["context_worse"] += 1
                improvement_category = "worse"
            else:
                results["summary"]["no_improvement"] += 1
                improvement_category = "no_change"
            
            comparison = {
                "query": test_query["query"],
                "context_response_length": context_length,
                "raw_response_length": raw_length,
                "context_entity_mentions": context_entity_mentions,
                "raw_entity_mentions": raw_entity_mentions,
                "improvement_score": improvement_score,
                "improvement_category": improvement_category,
                "context_time": context_time,
                "raw_time": raw_time
            }
            
            results["comparisons"].append(comparison)
            results["summary"]["avg_improvement"] += improvement_score
            
            logger.info(f"Context response: {context_length} chars, {context_entity_mentions} entity mentions")
            logger.info(f"Raw response: {raw_length} chars, {raw_entity_mentions} entity mentions")
            logger.info(f"Improvement: {improvement_score:.2f} ({improvement_category})")
        
        except Exception as e:
            logger.error(f"Error in RAG comparison for query {i+1}: {str(e)}")
            results["comparisons"].append({
                "query": test_query["query"],
                "error": str(e)
            })
    
    # Calculate average improvement
    if results["summary"]["total_tests"] > 0:
        results["summary"]["avg_improvement"] /= results["summary"]["total_tests"]
    
    return results

def generate_context_extraction_report(context_results: Dict, rag_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive context extraction report"""
    report = {
        "test_suite": "Context Extraction Tests",
        "execution_time": datetime.now().isoformat(),
        "context_extraction": context_results,
        "rag_comparison": rag_results,
        "summary": {
            "context_extraction_pass_rate": context_results["summary"]["passed"] / context_results["summary"]["total"] if context_results["summary"]["total"] > 0 else 0,
            "avg_context_quality": sum(context_results["summary"]["context_quality_scores"]) / len(context_results["summary"]["context_quality_scores"]) if context_results["summary"]["context_quality_scores"] else 0,
            "fast_route_percentage": context_results["summary"]["fast_route_count"] / context_results["summary"]["total"] if context_results["summary"]["total"] > 0 else 0,
            "context_improves_rag": rag_results["summary"]["context_improved"] > rag_results["summary"]["context_worse"]
        },
        "issues": context_results["issues"],
        "recommendations": []
    }
    
    # Generate recommendations
    if report["summary"]["context_extraction_pass_rate"] < 0.8:
        report["recommendations"].append({
            "priority": "high",
            "category": "context_extraction",
            "recommendation": f"Improve context extraction - only {report['summary']['context_extraction_pass_rate']:.1%} pass rate"
        })
    
    if report["summary"]["avg_context_quality"] < CONTEXT_QUALITY_THRESHOLD:
        report["recommendations"].append({
            "priority": "high",
            "category": "context_quality",
            "recommendation": f"Improve context quality - average score {report['summary']['avg_context_quality']:.2f} below threshold {CONTEXT_QUALITY_THRESHOLD}"
        })
    
    if report["summary"]["fast_route_percentage"] < FAST_ROUTE_TARGET:
        report["recommendations"].append({
            "priority": "medium",
            "category": "performance",
            "recommendation": f"Improve fast route usage - only {report['summary']['fast_route_percentage']:.1%} using fast patterns (target: {FAST_ROUTE_TARGET:.1%})"
        })
    
    if not report["summary"]["context_improves_rag"]:
        report["recommendations"].append({
            "priority": "high",
            "category": "rag_improvement",
            "recommendation": "Context-aware RAG not showing improvement over raw queries - check entity filtering and query enrichment"
        })
    
    return report

def print_report_summary(report: Dict):
    """Print formatted report summary to console"""
    print("\n" + "="*80)
    print("CONTEXT EXTRACTION TEST RESULTS")
    print("="*80)
    
    context_summary = report["context_extraction"]["summary"]
    print(f"\nContext Extraction Summary:")
    print(f"  Total Test Cases: {context_summary['total']}")
    print(f"  Passed: {context_summary['passed']} ")
    print(f"  Failed: {context_summary['failed']} ")
    print(f"  Pass Rate: {report['summary']['context_extraction_pass_rate']:.1%}")
    print(f"  Average Context Quality: {report['summary']['avg_context_quality']:.2f}")
    print(f"  Fast Route Usage: {report['summary']['fast_route_percentage']:.1%} (target: {FAST_ROUTE_TARGET:.1%})")
    
    rag_summary = report["rag_comparison"]["summary"]
    print(f"\nRAG Comparison Summary:")
    print(f"  Total Comparisons: {rag_summary['total_tests']}")
    print(f"  Context Improved: {rag_summary['context_improved']} ")
    print(f"  No Change: {rag_summary['no_improvement']} ")
    print(f"  Context Worse: {rag_summary['context_worse']} ")
    print(f"  Average Improvement: {rag_summary['avg_improvement']:.2f}")
    
    if report["issues"]:
        print(f"\nIssues Found ({len(report['issues'])}):")
        for issue in report["issues"][:5]:  # Show first 5 issues
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
    logger.info("Starting Leibniz Context Extraction Tests")
    
    # Create test cases
    test_cases = create_context_test_cases()
    
    # Test context extraction
    context_results = await test_context_extraction(test_cases)
    
    # Test context-aware RAG comparison
    rag_results = await test_context_aware_rag_comparison()
    
    # Generate report
    report = generate_context_extraction_report(context_results, rag_results)
    
    # Save results
    output_dir = Path("leibniz_agent/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "context_extraction_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save markdown report
    with open(output_dir / "CONTEXT_EXTRACTION_REPORT.md", "w") as f:
        f.write("# Context Extraction Test Report\n\n")
        f.write(f"Generated: {report['execution_time']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Context Extraction Pass Rate: {report['summary']['context_extraction_pass_rate']:.1%}\n")
        f.write(f"- Average Context Quality: {report['summary']['avg_context_quality']:.2f}\n")
        f.write(f"- Fast Route Usage: {report['summary']['fast_route_percentage']:.1%}\n")
        f.write(f"- Context Improves RAG: {'Yes' if report['summary']['context_improves_rag'] else 'No'}\n\n")
        
        f.write("## Context Extraction Results\n\n")
        for test_case in report['context_extraction']['test_cases']:
            f.write(f"### {test_case['name']}\n")
            f.write(f"- Input: `{test_case['input']}`\n")
            f.write(f"- Expected Intent: {test_case.get('expected_intent', 'N/A')}\n")
            f.write(f"- Actual Intent: {test_case.get('actual_intent', 'N/A')}\n")
            if 'context_validation' in test_case:
                f.write(f"- Context Quality: {test_case['context_validation']['overall_score']:.2f}\n")
                f.write(f"- Fast Route: {'Yes' if test_case.get('fast_route', False) else 'No'}\n")
            f.write(f"- Result: {'PASS' if test_case.get('passed', False) else 'FAIL'}\n\n")
        
        f.write("## RAG Comparison Results\n\n")
        for comparison in report['rag_comparison']['comparisons']:
            if 'error' not in comparison:
                f.write(f"### Query: {comparison['query']}\n")
                f.write(f"- Context Response: {comparison['context_response_length']} chars\n")
                f.write(f"- Raw Response: {comparison['raw_response_length']} chars\n")
                f.write(f"- Improvement Score: {comparison['improvement_score']:.2f}\n")
                f.write(f"- Category: {comparison['improvement_category']}\n\n")
        
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
    return 0 if report["summary"]["context_extraction_pass_rate"] >= 0.8 and report["summary"]["context_improves_rag"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
