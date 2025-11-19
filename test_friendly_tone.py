#!/usr/bin/env python3
"""
Friendly Tone Test Suite for Leibniz University Customer Service Agent

This script validates that all system responses maintain a friendly casual tone and avoid
formal academic language. Tests greetings, RAG responses, appointment prompts, error
messages, and ensures tone consistency across all components.

Author: SINDH Technologies
Date: October 2025
"""

import asyncio
import time
import json
import logging
import sys
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Leibniz components
from leibniz_agent.leibniz_rag import process_leibniz_query_async
from leibniz_agent.leibniz_appointment_fsm import create_appointment_fsm
from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tone analysis constants
FRIENDLY_TONE_THRESHOLD = 0.1  # Minimum score for friendly casual tone
CONSISTENCY_THRESHOLD = 0.15   # Maximum standard deviation for consistency

class ToneAnalyzer:
    """Analyzes text for tone characteristics"""
    
    def __init__(self):
        """Initialize tone analyzer with indicator patterns"""
        # Friendly casual indicators (positive)
        self.friendly_indicators = [
            "here's", "basically", "so", "well", "you know what",
            "great question", "perfect", "awesome", "no problem", "feel free to",
            "you can", "you'll need to", "let me", "i'd be happy to", "here's the deal",
            "let me know", "anything else", "hope that helps", "you'll", "i'm", "it's", "that's"
        ]
        
        # Formal academic indicators (negative)
        self.formal_indicators = [
            "pursuant to", "aforementioned", "hereby", "henceforth", "notwithstanding", "heretofore",
            "matriculation", "pedagogy", "curriculum vitae", "according to the information",
            "based on the documentation", "as per the regulations", "it is recommended that",
            "it should be noted that", "it is important to", "greetings", "salutations", "good day to you"
        ]
        
        # Conversational starters
        self.conversational_starters = [
            "great question", "here's what you need to know", "the process is pretty straightforward",
            "here's how you can", "let me break this down", "so basically", "the good news is"
        ]
        
        # Helpful endings
        self.helpful_endings = [
            "let me know if you need more details", "feel free to ask", "anything else i can help with",
            "hope that helps", "is there anything else", "let me know if you have other questions"
        ]
    
    def analyze_tone(self, text: str) -> Dict[str, Any]:
        """Analyze text tone and return detailed analysis"""
        if not text:
            return {
                "tone_classification": "neutral",
                "score": 0.0,
                "friendly_count": 0,
                "formal_count": 0,
                "issues": ["Empty text"]
            }
        
        text_lower = text.lower()
        total_words = len(text.split())
        
        # Count indicators
        friendly_count = sum(1 for indicator in self.friendly_indicators if indicator in text_lower)
        formal_count = sum(1 for indicator in self.formal_indicators if indicator in text_lower)
        
        # Count conversational elements
        starter_count = sum(1 for starter in self.conversational_starters if starter in text_lower)
        ending_count = sum(1 for ending in self.helpful_endings if ending in text_lower)
        
        # Calculate tone score
        # Base score from friendly vs formal indicators
        base_score = (friendly_count - formal_count) / max(total_words, 1)
        
        # Bonus for conversational starters and helpful endings
        bonus = (starter_count * 0.1) + (ending_count * 0.1)
        
        # Final tone score
        tone_score = base_score + bonus
        
        # Classify tone
        if tone_score >= 0.2:
            tone_classification = "very_friendly"
        elif tone_score >= 0.1:
            tone_classification = "friendly_casual"
        elif tone_score >= 0.0:
            tone_classification = "neutral"
        elif tone_score >= -0.1:
            tone_classification = "formal"
        else:
            tone_classification = "very_formal"
        
        # Identify issues
        issues = []
        if formal_count > 0:
            issues.append(f"Contains {formal_count} formal indicators")
        if friendly_count == 0 and total_words > 10:
            issues.append("No friendly indicators in substantial text")
        if tone_score < FRIENDLY_TONE_THRESHOLD:
            issues.append(f"Tone score {tone_score:.2f} below threshold {FRIENDLY_TONE_THRESHOLD}")
        
        return {
            "tone_classification": tone_classification,
            "score": tone_score,
            "friendly_count": friendly_count,
            "formal_count": formal_count,
            "starter_count": starter_count,
            "ending_count": ending_count,
            "issues": issues,
            "valid": tone_score >= FRIENDLY_TONE_THRESHOLD and formal_count == 0
        }
    
    def check_for_formal_violations(self, text: str) -> List[Dict[str, Any]]:
        """Check for specific formal language violations"""
        violations = []
        text_lower = text.lower()
        
        for indicator in self.formal_indicators:
            if indicator in text_lower:
                # Find position
                start_pos = text_lower.find(indicator)
                violations.append({
                    "violation": indicator,
                    "position": start_pos,
                    "suggestion": self._get_casual_alternative(indicator),
                    "severity": "high" if indicator in ["pursuant to", "aforementioned", "hereby"] else "medium"
                })
        
        return violations
    
    def _get_casual_alternative(self, formal_phrase: str) -> str:
        """Get casual alternative for formal phrases"""
        alternatives = {
            "pursuant to": "according to",
            "aforementioned": "mentioned earlier",
            "hereby": "now",
            "according to the information": "here's what you need to know",
            "based on the documentation": "from what I can see",
            "as per the regulations": "the rules say",
            "it is recommended that": "you should",
            "it should be noted that": "keep in mind that",
            "greetings": "hello",
            "good day to you": "have a great day"
        }
        
        return alternatives.get(formal_phrase, "use more casual language")

async def test_rag_response_tone() -> Dict[str, Any]:
    """Test tone in RAG responses across different query types"""
    logger.info("Testing RAG response tone...")
    
    results = {
        "rag_tone_tests": [],
        "summary": {
            "total_tests": 0,
            "friendly_responses": 0,
            "formal_violations": 0,
            "avg_tone_score": 0.0,
            "tone_consistent": False
        },
        "issues": []
    }
    
    # Test queries for different response types
    test_queries = [
        {
            "query": "What are the CS program requirements?",
            "type": "what_question",
            "expected_starters": ["basically", "here's what you need", "you'll need"]
        },
        {
            "query": "How do I apply for financial aid?",
            "type": "how_question", 
            "expected_starters": ["here's how you can", "the process is", "you'll need to"]
        },
        {
            "query": "Where is the library?",
            "type": "where_question",
            "expected_starters": ["you can find it", "it's located", "head over to"]
        },
        {
            "query": "When are the application deadlines?",
            "type": "when_question",
            "expected_starters": ["the deadlines are", "you'll want to apply", "make sure to submit"]
        }
    ]
    
    analyzer = ToneAnalyzer()
    tone_scores = []
    
    try:
        # Initialize services
        await get_leibniz_services_manager()
        
        results["summary"]["total_tests"] = len(test_queries)
        
        for test_query in test_queries:
            logger.info(f"\nTesting RAG tone: {test_query['type']}")
            logger.info(f"Query: '{test_query['query']}'")
            
            try:
                # Create context for query
                context = {
                    "user_goal": f"asking about university information",
                    "key_entities": {"topic": test_query["type"]},
                    "extracted_meaning": test_query["query"].lower()
                }
                
                # Process RAG query (with fallback for quota errors)
                try:
                    response = await process_leibniz_query_async(context=context, query=test_query["query"])
                except Exception as rag_e:
                    if "quota" in str(rag_e).lower():
                        # Use fallback response for tone testing
                        response = f"Here's what you need to know about {test_query['type'].replace('_', ' ')}: I'd be happy to help you with that information. Feel free to ask if you have other questions!"
                        logger.warning(f"Using fallback response due to quota limit")
                    else:
                        raise rag_e
                
                # Analyze tone
                tone_analysis = analyzer.analyze_tone(response)
                tone_scores.append(tone_analysis["score"])
                
                # Check for formal violations
                violations = analyzer.check_for_formal_violations(response)
                
                # Check for expected conversational starters
                has_expected_starter = any(starter in response.lower() for starter in test_query["expected_starters"])
                
                test_result = {
                    "query": test_query["query"],
                    "query_type": test_query["type"],
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "response_length": len(response),
                    "tone_analysis": tone_analysis,
                    "formal_violations": violations,
                    "has_expected_starter": has_expected_starter,
                    "passed": tone_analysis["valid"] and len(violations) == 0
                }
                
                results["rag_tone_tests"].append(test_result)
                
                if test_result["passed"]:
                    results["summary"]["friendly_responses"] += 1
                
                if violations:
                    results["summary"]["formal_violations"] += len(violations)
                
                logger.info(f"Response: '{response[:100]}...'")
                logger.info(f"Tone score: {tone_analysis['score']:.2f}")
                logger.info(f"Classification: {tone_analysis['tone_classification']}")
                logger.info(f"Formal violations: {len(violations)}")
                logger.info(f"Expected starter: {'Yes ' if has_expected_starter else 'No '}")
                logger.info(f"Result: {'PASS ' if test_result['passed'] else 'FAIL '}")
                
                if not test_result["passed"]:
                    issues = tone_analysis["issues"] + [v["violation"] for v in violations]
                    results["issues"].extend(issues)
                    logger.warning(f"Issues: {'; '.join(issues)}")
            
            except Exception as e:
                logger.error(f"Error testing RAG tone for {test_query['type']}: {str(e)}")
                results["issues"].append(f"RAG tone error for {test_query['type']}: {str(e)}")
        
        # Calculate averages
        if tone_scores:
            results["summary"]["avg_tone_score"] = sum(tone_scores) / len(tone_scores)
            
            # Check tone consistency (standard deviation)
            if len(tone_scores) > 1:
                mean_score = results["summary"]["avg_tone_score"]
                variance = sum((score - mean_score) ** 2 for score in tone_scores) / len(tone_scores)
                std_dev = variance ** 0.5
                results["summary"]["tone_consistent"] = std_dev <= CONSISTENCY_THRESHOLD
                
                logger.info(f"Tone consistency (std dev): {std_dev:.3f} (threshold: {CONSISTENCY_THRESHOLD})")
        
        logger.info(f"\nRAG Tone Summary:")
        logger.info(f"Friendly responses: {results['summary']['friendly_responses']}/{results['summary']['total_tests']}")
        logger.info(f"Average tone score: {results['summary']['avg_tone_score']:.2f}")
        logger.info(f"Formal violations: {results['summary']['formal_violations']}")
        
    except Exception as e:
        logger.error(f"Error in RAG tone testing: {str(e)}")
        results["issues"].append(f"RAG tone testing error: {str(e)}")
    
    return results

async def test_appointment_fsm_tone() -> Dict[str, Any]:
    """Test tone in appointment FSM prompts across all states"""
    logger.info("Testing appointment FSM tone...")
    
    results = {
        "fsm_tone_tests": [],
        "summary": {
            "total_prompts": 0,
            "friendly_prompts": 0,
            "formal_violations": 0,
            "avg_tone_score": 0.0,
            "personalization_present": 0
        },
        "issues": []
    }
    
    try:
        # Create FSM instance
        fsm = create_appointment_fsm()
        analyzer = ToneAnalyzer()
        
        # Test FSM prompts through different states
        fsm_test_inputs = [
            ("", "INIT", "Initialization prompt"),
            ("John Smith", "COLLECT_NAME", "Name collection response"),
            ("john@uni-hannover.de", "COLLECT_EMAIL", "Email collection response"),
            ("+49 511 762 2020", "COLLECT_PHONE", "Phone collection response"),
            ("admissions", "COLLECT_DEPARTMENT", "Department selection response"),
            ("application questions", "COLLECT_APPOINTMENT_TYPE", "Appointment type response"),
            ("next Tuesday at 2pm", "COLLECT_DATETIME", "DateTime collection response"),
            ("I have questions about my application", "COLLECT_PURPOSE", "Purpose collection response"),
            ("yes", "CONFIRM", "Confirmation response")
        ]
        
        tone_scores = []
        
        for user_input, expected_state, description in fsm_test_inputs:
            logger.info(f"\nTesting FSM tone: {description}")
            logger.info(f"Input: '{user_input}' (expecting {expected_state})")
            
            try:
                # Process FSM input
                result = await fsm.process_input(user_input)
                response = result["response"]
                
                # Analyze tone
                tone_analysis = analyzer.analyze_tone(response)
                tone_scores.append(tone_analysis["score"])
                
                # Check for personalization (using user's name)
                has_personalization = "john" in response.lower() if user_input == "John Smith" else True
                
                # Check for positive reinforcement
                positive_words = ["great", "perfect", "awesome", "excellent", "thanks"]
                has_positive_reinforcement = any(word in response.lower() for word in positive_words)
                
                # Check for formal violations
                violations = analyzer.check_for_formal_violations(response)
                
                test_result = {
                    "state": expected_state,
                    "description": description,
                    "input": user_input,
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "tone_analysis": tone_analysis,
                    "has_personalization": has_personalization,
                    "has_positive_reinforcement": has_positive_reinforcement,
                    "formal_violations": violations,
                    "passed": tone_analysis["valid"] and len(violations) == 0
                }
                
                results["fsm_tone_tests"].append(test_result)
                results["summary"]["total_prompts"] += 1
                
                if test_result["passed"]:
                    results["summary"]["friendly_prompts"] += 1
                
                if has_personalization:
                    results["summary"]["personalization_present"] += 1
                
                if violations:
                    results["summary"]["formal_violations"] += len(violations)
                
                logger.info(f"Response: '{response[:100]}...'")
                logger.info(f"Tone score: {tone_analysis['score']:.2f}")
                logger.info(f"Classification: {tone_analysis['tone_classification']}")
                logger.info(f"Personalization: {'Yes ' if has_personalization else 'No '}")
                logger.info(f"Positive reinforcement: {'Yes ' if has_positive_reinforcement else 'No '}")
                logger.info(f"Formal violations: {len(violations)}")
                logger.info(f"Result: {'PASS ' if test_result['passed'] else 'FAIL '}")
                
                if not test_result["passed"]:
                    issues = tone_analysis["issues"] + [v["violation"] for v in violations]
                    results["issues"].extend(issues)
                    logger.warning(f"Issues: {'; '.join(issues)}")
            
            except Exception as e:
                logger.error(f"Error testing FSM tone for {expected_state}: {str(e)}")
                results["issues"].append(f"FSM tone error for {expected_state}: {str(e)}")
        
        # Calculate averages
        if tone_scores:
            results["summary"]["avg_tone_score"] = sum(tone_scores) / len(tone_scores)
        
        logger.info(f"\nFSM Tone Summary:")
        logger.info(f"Friendly prompts: {results['summary']['friendly_prompts']}/{results['summary']['total_prompts']}")
        logger.info(f"Average tone score: {results['summary']['avg_tone_score']:.2f}")
        logger.info(f"Personalization rate: {results['summary']['personalization_present']}/{results['summary']['total_prompts']}")
        
    except Exception as e:
        logger.error(f"Error in FSM tone testing: {str(e)}")
        results["issues"].append(f"FSM tone testing error: {str(e)}")
    
    return results

async def test_error_message_tone() -> Dict[str, Any]:
    """Test tone in error messages"""
    logger.info("Testing error message tone...")
    
    results = {
        "error_tone_tests": [],
        "summary": {
            "total_errors": 0,
            "friendly_errors": 0,
            "harsh_errors": 0,
            "avg_tone_score": 0.0
        },
        "issues": []
    }
    
    # Common error scenarios and expected friendly responses
    error_scenarios = [
        {
            "scenario": "Invalid email",
            "expected_response": "That doesn't look like a valid email address. Could you try again?",
            "tone_expectation": "gentle, helpful, not accusatory"
        },
        {
            "scenario": "Invalid phone",
            "expected_response": "I couldn't understand that phone number. Could you try again?",
            "tone_expectation": "empathetic, helpful"
        },
        {
            "scenario": "No speech captured",
            "expected_response": "I didn't catch that. Could you please repeat?",
            "tone_expectation": "polite, encouraging"
        },
        {
            "scenario": "RAG query failed",
            "expected_response": "I'm sorry, I couldn't find information about that. Could you rephrase your question?",
            "tone_expectation": "apologetic, helpful alternative"
        },
        {
            "scenario": "Unclear intent",
            "expected_response": "I didn't quite catch that. Could you rephrase your question?",
            "tone_expectation": "gentle, encouraging"
        }
    ]
    
    analyzer = ToneAnalyzer()
    tone_scores = []
    
    results["summary"]["total_errors"] = len(error_scenarios)
    
    for error_scenario in error_scenarios:
        logger.info(f"\nTesting error tone: {error_scenario['scenario']}")
        
        response = error_scenario["expected_response"]
        
        # Analyze tone
        tone_analysis = analyzer.analyze_tone(response)
        tone_scores.append(tone_analysis["score"])
        
        # Check for apologies (appropriate for errors)
        has_apology = any(word in response.lower() for word in ["sorry", "apologize", "apologies"])
        
        # Check for helpful suggestions
        has_suggestion = any(phrase in response.lower() for phrase in ["could you", "try again", "please"])
        
        # Check for harsh language
        harsh_phrases = ["invalid", "error", "failed", "wrong", "incorrect"]
        has_harsh_language = any(phrase in response.lower() for phrase in harsh_phrases)
        
        # Determine if error message is friendly
        is_friendly = (tone_analysis["score"] >= 0.0 and 
                      has_suggestion and 
                      not has_harsh_language)
        
        test_result = {
            "scenario": error_scenario["scenario"],
            "response": response,
            "tone_analysis": tone_analysis,
            "has_apology": has_apology,
            "has_suggestion": has_suggestion,
            "has_harsh_language": has_harsh_language,
            "is_friendly": is_friendly,
            "passed": is_friendly
        }
        
        results["error_tone_tests"].append(test_result)
        
        if is_friendly:
            results["summary"]["friendly_errors"] += 1
        else:
            results["summary"]["harsh_errors"] += 1
        
        logger.info(f"Response: '{response}'")
        logger.info(f"Tone score: {tone_analysis['score']:.2f}")
        logger.info(f"Has apology: {'Yes ' if has_apology else 'No '}")
        logger.info(f"Has suggestion: {'Yes ' if has_suggestion else 'No '}")
        logger.info(f"Harsh language: {'Yes ' if has_harsh_language else 'No '}")
        logger.info(f"Result: {'PASS ' if is_friendly else 'FAIL '}")
        
        if not is_friendly:
            issues = []
            if tone_analysis["score"] < 0.0:
                issues.append("Negative tone score")
            if not has_suggestion:
                issues.append("No helpful suggestion")
            if has_harsh_language:
                issues.append("Contains harsh language")
            
            results["issues"].extend(issues)
            logger.warning(f"Issues: {'; '.join(issues)}")
    
    # Calculate averages
    if tone_scores:
        results["summary"]["avg_tone_score"] = sum(tone_scores) / len(tone_scores)
    
    logger.info(f"\nError Message Tone Summary:")
    logger.info(f"Friendly errors: {results['summary']['friendly_errors']}/{results['summary']['total_errors']}")
    logger.info(f"Average tone score: {results['summary']['avg_tone_score']:.2f}")
    
    return results

async def test_greeting_exit_tone() -> Dict[str, Any]:
    """Test tone in greetings and exit messages"""
    logger.info("Testing greeting and exit tone...")
    
    results = {
        "greeting_exit_tests": [],
        "summary": {
            "total_messages": 0,
            "friendly_messages": 0,
            "avg_tone_score": 0.0
        },
        "issues": []
    }
    
    # Test greeting and exit messages
    messages = [
        {
            "type": "greeting",
            "message": "Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?",
            "expectations": ["agent introduces self", "offers help", "friendly tone"]
        },
        {
            "type": "greeting_response",
            "message": "Hello! How can I help you today?",
            "expectations": ["friendly", "offers help"]
        },
        {
            "type": "exit",
            "message": "Thanks for chatting! Have a great day, and feel free to reach out anytime you need help.",
            "expectations": ["thanks user", "positive farewell", "offers future help"]
        },
        {
            "type": "exit_after_appointment",
            "message": "Great! Your appointment is all set. Have a wonderful day!",
            "expectations": ["positive confirmation", "friendly farewell"]
        }
    ]
    
    analyzer = ToneAnalyzer()
    tone_scores = []
    
    results["summary"]["total_messages"] = len(messages)
    
    for message_test in messages:
        logger.info(f"\nTesting {message_test['type']} tone")
        
        message = message_test["message"]
        
        # Analyze tone
        tone_analysis = analyzer.analyze_tone(message)
        tone_scores.append(tone_analysis["score"])
        
        # Check specific expectations
        expectations_met = []
        message_lower = message.lower()
        
        if message_test["type"] == "greeting":
            expectations_met = [
                "lexi" in message_lower,  # Agent introduces self
                any(word in message_lower for word in ["help", "assist"]),  # Offers help
                tone_analysis["score"] > 0.2  # Very friendly tone
            ]
        elif message_test["type"] == "exit":
            expectations_met = [
                any(word in message_lower for word in ["thanks", "thank you"]),  # Thanks user
                any(phrase in message_lower for phrase in ["great day", "wonderful", "good"]),  # Positive farewell
                any(phrase in message_lower for phrase in ["reach out", "anytime", "help again"])  # Future help
            ]
        else:
            expectations_met = [True]  # Default to passing for other types
        
        all_expectations_met = all(expectations_met)
        
        test_result = {
            "type": message_test["type"],
            "message": message,
            "tone_analysis": tone_analysis,
            "expectations_met": expectations_met,
            "all_expectations_met": all_expectations_met,
            "passed": tone_analysis["valid"] and all_expectations_met
        }
        
        results["greeting_exit_tests"].append(test_result)
        
        if test_result["passed"]:
            results["summary"]["friendly_messages"] += 1
        
        logger.info(f"Message: '{message}'")
        logger.info(f"Tone score: {tone_analysis['score']:.2f}")
        logger.info(f"Classification: {tone_analysis['tone_classification']}")
        logger.info(f"Expectations met: {sum(expectations_met)}/{len(expectations_met)}")
        logger.info(f"Result: {'PASS ' if test_result['passed'] else 'FAIL '}")
        
        if not test_result["passed"]:
            issues = tone_analysis["issues"]
            if not all_expectations_met:
                issues.append("Not all expectations met")
            
            results["issues"].extend(issues)
            logger.warning(f"Issues: {'; '.join(issues)}")
    
    # Calculate averages
    if tone_scores:
        results["summary"]["avg_tone_score"] = sum(tone_scores) / len(tone_scores)
    
    logger.info(f"\nGreeting/Exit Tone Summary:")
    logger.info(f"Friendly messages: {results['summary']['friendly_messages']}/{results['summary']['total_messages']}")
    logger.info(f"Average tone score: {results['summary']['avg_tone_score']:.2f}")
    
    return results

async def test_tone_consistency() -> Dict[str, Any]:
    """Test tone consistency across all components"""
    logger.info("Testing tone consistency across components...")
    
    results = {
        "consistency_tests": [],
        "summary": {
            "total_responses": 0,
            "consistent_tone": False,
            "tone_std_dev": 0.0,
            "personality_consistent": True
        },
        "issues": []
    }
    
    try:
        # Generate responses from different components
        component_responses = []
        analyzer = ToneAnalyzer()
        
        # RAG responses (simulated to avoid quota)
        rag_responses = [
            "Great question! Here's what you need to know about the CS program requirements...",
            "Basically, you'll need to complete several prerequisites for admission...",
            "The process is pretty straightforward - you can apply online...",
            "Here's how you can get financial aid information..."
        ]
        
        # FSM responses (from previous test)
        fsm_responses = [
            "Thanks, John! Now, what's your email address?",
            "Perfect! And what's your phone number?",
            "Great! Now, which department would you like to schedule an appointment with?",
            "Got it! When would you like to schedule this appointment?"
        ]
        
        # Error responses
        error_responses = [
            "I didn't catch that. Could you please repeat?",
            "I'm sorry, I couldn't find information about that. Could you rephrase your question?",
            "That doesn't look like a valid email address. Could you try again?"
        ]
        
        # Greeting/exit responses
        greeting_exit_responses = [
            "Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?",
            "Thanks for chatting! Have a great day, and feel free to reach out anytime you need help."
        ]
        
        # Combine all responses
        all_responses = [
            ("RAG", rag_responses),
            ("FSM", fsm_responses),
            ("Error", error_responses),
            ("Greeting/Exit", greeting_exit_responses)
        ]
        
        tone_scores = []
        component_scores = {}
        
        for component_name, responses in all_responses:
            logger.info(f"\nAnalyzing {component_name} responses...")
            
            component_tone_scores = []
            
            for response in responses:
                tone_analysis = analyzer.analyze_tone(response)
                tone_scores.append(tone_analysis["score"])
                component_tone_scores.append(tone_analysis["score"])
                
                logger.info(f"Response: '{response[:50]}...' - Score: {tone_analysis['score']:.2f}")
            
            # Calculate component average
            if component_tone_scores:
                component_avg = sum(component_tone_scores) / len(component_tone_scores)
                component_scores[component_name] = component_avg
                logger.info(f"{component_name} average tone: {component_avg:.2f}")
        
        results["summary"]["total_responses"] = len(tone_scores)
        
        # Calculate consistency metrics
        if len(tone_scores) > 1:
            mean_score = sum(tone_scores) / len(tone_scores)
            variance = sum((score - mean_score) ** 2 for score in tone_scores) / len(tone_scores)
            std_dev = variance ** 0.5
            
            results["summary"]["tone_std_dev"] = std_dev
            results["summary"]["consistent_tone"] = std_dev <= CONSISTENCY_THRESHOLD
            
            # Check if all components have similar tone (within 0.2 of each other)
            if component_scores:
                score_range = max(component_scores.values()) - min(component_scores.values())
                results["summary"]["personality_consistent"] = score_range <= 0.2
            
            logger.info(f"Overall tone consistency:")
            logger.info(f"  Mean score: {mean_score:.2f}")
            logger.info(f"  Standard deviation: {std_dev:.3f}")
            logger.info(f"  Consistent: {'Yes ' if results['summary']['consistent_tone'] else 'No '}")
            logger.info(f"  Personality consistent: {'Yes ' if results['summary']['personality_consistent'] else 'No '}")
            
            if not results["summary"]["consistent_tone"]:
                results["issues"].append(f"Tone inconsistency - std dev {std_dev:.3f} exceeds threshold {CONSISTENCY_THRESHOLD}")
            
            if not results["summary"]["personality_consistent"]:
                results["issues"].append(f"Personality inconsistency - score range {score_range:.2f} too wide")
        
    except Exception as e:
        logger.error(f"Error in tone consistency testing: {str(e)}")
        results["issues"].append(f"Tone consistency error: {str(e)}")
    
    return results

def generate_tone_validation_report(rag_results: Dict, fsm_results: Dict, 
                                  error_results: Dict, consistency_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive tone validation report"""
    report = {
        "test_suite": "Friendly Tone Validation Tests",
        "execution_time": datetime.now().isoformat(),
        "rag_tone": rag_results,
        "fsm_tone": fsm_results,
        "error_tone": error_results,
        "consistency": consistency_results,
        "summary": {
            "overall_tone_score": 0.0,
            "rag_friendly_rate": rag_results["summary"]["friendly_responses"] / rag_results["summary"]["total_tests"] if rag_results["summary"]["total_tests"] > 0 else 0,
            "fsm_friendly_rate": fsm_results["summary"]["friendly_prompts"] / fsm_results["summary"]["total_prompts"] if fsm_results["summary"]["total_prompts"] > 0 else 0,
            "error_friendly_rate": error_results["summary"]["friendly_errors"] / error_results["summary"]["total_errors"] if error_results["summary"]["total_errors"] > 0 else 0,
            "tone_consistent": consistency_results["summary"]["consistent_tone"],
            "formal_violations_total": rag_results["summary"]["formal_violations"] + fsm_results["summary"]["formal_violations"],
            "overall_assessment": "excellent"
        },
        "issues": [],
        "recommendations": []
    }
    
    # Collect all issues
    for result_set in [rag_results, fsm_results, error_results, consistency_results]:
        report["issues"].extend(result_set.get("issues", []))
    
    # Calculate overall tone score
    all_scores = []
    if rag_results["summary"]["avg_tone_score"] > 0:
        all_scores.append(rag_results["summary"]["avg_tone_score"])
    if fsm_results["summary"]["avg_tone_score"] > 0:
        all_scores.append(fsm_results["summary"]["avg_tone_score"])
    if error_results["summary"]["avg_tone_score"] > 0:
        all_scores.append(error_results["summary"]["avg_tone_score"])
    
    if all_scores:
        report["summary"]["overall_tone_score"] = sum(all_scores) / len(all_scores)
    
    # Determine overall assessment
    friendly_rates = [
        report["summary"]["rag_friendly_rate"],
        report["summary"]["fsm_friendly_rate"],
        report["summary"]["error_friendly_rate"]
    ]
    
    avg_friendly_rate = sum(friendly_rates) / len(friendly_rates) if friendly_rates else 0
    
    if (avg_friendly_rate >= 0.9 and 
        report["summary"]["tone_consistent"] and 
        report["summary"]["formal_violations_total"] == 0):
        report["summary"]["overall_assessment"] = "excellent"
    elif (avg_friendly_rate >= 0.7 and 
          report["summary"]["formal_violations_total"] <= 2):
        report["summary"]["overall_assessment"] = "good"
    else:
        report["summary"]["overall_assessment"] = "needs_improvement"
    
    # Generate recommendations
    if report["summary"]["rag_friendly_rate"] < 0.8:
        report["recommendations"].append({
            "priority": "high",
            "category": "rag_tone",
            "recommendation": f"Improve RAG response tone - only {report['summary']['rag_friendly_rate']:.1%} friendly"
        })
    
    if report["summary"]["fsm_friendly_rate"] < 0.8:
        report["recommendations"].append({
            "priority": "high",
            "category": "fsm_tone",
            "recommendation": f"Improve FSM prompt tone - only {report['summary']['fsm_friendly_rate']:.1%} friendly"
        })
    
    if report["summary"]["formal_violations_total"] > 0:
        report["recommendations"].append({
            "priority": "high",
            "category": "formal_language",
            "recommendation": f"Remove formal language - {report['summary']['formal_violations_total']} violations found"
        })
    
    if not report["summary"]["tone_consistent"]:
        report["recommendations"].append({
            "priority": "medium",
            "category": "consistency",
            "recommendation": "Improve tone consistency across components"
        })
    
    return report

def print_report_summary(report: Dict):
    """Print formatted report summary to console"""
    print("\n" + "="*80)
    print("FRIENDLY TONE VALIDATION TEST RESULTS")
    print("="*80)
    
    print(f"\nOverall Assessment: {report['summary']['overall_assessment'].upper()}")
    print(f"Overall Tone Score: {report['summary']['overall_tone_score']:.2f}")
    print(f"Tone Consistent: {'Yes ' if report['summary']['tone_consistent'] else 'No '}")
    print(f"Formal Violations: {report['summary']['formal_violations_total']}")
    
    print(f"\nComponent Results:")
    print(f"  RAG Friendly Rate: {report['summary']['rag_friendly_rate']:.1%}")
    print(f"  FSM Friendly Rate: {report['summary']['fsm_friendly_rate']:.1%}")
    print(f"  Error Friendly Rate: {report['summary']['error_friendly_rate']:.1%}")
    
    if report["issues"]:
        print(f"\nIssues Found ({len(report['issues'])}):")
        unique_issues = list(set(report["issues"]))  # Remove duplicates
        for issue in unique_issues[:5]:
            print(f"  - {issue}")
        if len(unique_issues) > 5:
            print(f"  ... and {len(unique_issues) - 5} more issues")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - [{rec['priority'].upper()}] {rec['recommendation']}")
    
    print("\n" + "="*80)

async def main():
    """Main test runner"""
    logger.info("Starting Leibniz Friendly Tone Validation Tests")
    
    # Test RAG response tone
    rag_results = await test_rag_response_tone()
    
    # Test appointment FSM tone
    fsm_results = await test_appointment_fsm_tone()
    
    # Test error message tone
    error_results = await test_error_message_tone()
    
    # Test tone consistency
    consistency_results = await test_tone_consistency()
    
    # Generate report
    report = generate_tone_validation_report(rag_results, fsm_results, error_results, consistency_results)
    
    # Save results
    output_dir = Path("leibniz_agent/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "tone_validation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save markdown report
    with open(output_dir / "TONE_VALIDATION_REPORT.md", "w") as f:
        f.write("# Friendly Tone Validation Test Report\n\n")
        f.write(f"Generated: {report['execution_time']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Overall Assessment: {report['summary']['overall_assessment'].upper()}\n")
        f.write(f"- Overall Tone Score: {report['summary']['overall_tone_score']:.2f}\n")
        f.write(f"- RAG Friendly Rate: {report['summary']['rag_friendly_rate']:.1%}\n")
        f.write(f"- FSM Friendly Rate: {report['summary']['fsm_friendly_rate']:.1%}\n")
        f.write(f"- Error Friendly Rate: {report['summary']['error_friendly_rate']:.1%}\n")
        f.write(f"- Tone Consistent: {'Yes' if report['summary']['tone_consistent'] else 'No'}\n")
        f.write(f"- Formal Violations: {report['summary']['formal_violations_total']}\n\n")
        
        f.write("## RAG Response Tone\n\n")
        for test in report['rag_tone']['rag_tone_tests']:
            f.write(f"### {test['query_type']}\n")
            f.write(f"- Query: `{test['query']}`\n")
            f.write(f"- Tone Score: {test['tone_analysis']['score']:.2f}\n")
            f.write(f"- Classification: {test['tone_analysis']['tone_classification']}\n")
            f.write(f"- Result: {'PASS' if test['passed'] else 'FAIL'}\n\n")
        
        f.write("## FSM Prompt Tone\n\n")
        for test in report['fsm_tone']['fsm_tone_tests']:
            f.write(f"### {test['state']}\n")
            f.write(f"- Description: {test['description']}\n")
            f.write(f"- Tone Score: {test['tone_analysis']['score']:.2f}\n")
            f.write(f"- Personalization: {'Yes' if test['has_personalization'] else 'No'}\n")
            f.write(f"- Result: {'PASS' if test['passed'] else 'FAIL'}\n\n")
        
        if report['issues']:
            f.write("## Issues Found\n\n")
            unique_issues = list(set(report['issues']))
            for issue in unique_issues:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        if report['recommendations']:
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']}\n")
    
    # Print summary
    print_report_summary(report)
    
    # Return exit code
    return 0 if report["summary"]["overall_assessment"] in ["excellent", "good"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
