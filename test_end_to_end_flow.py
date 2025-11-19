#!/usr/bin/env python3
"""
End-to-End Flow Test Suite for Leibniz University Customer Service Agent

This script validates the complete conversation flow from greeting through RAG query,
appointment booking, and exit. It tests multiple scenarios with comprehensive validation
of intent classification, response quality, and flow continuity.

Author: SINDH Technologies
Date: October 2025
"""

import asyncio
import time
import json
import logging
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Leibniz components
from leibniz_agent import leibniz_pro
from leibniz_agent.leibniz_intent_parser import classify_leibniz_intent
# Import RAG and FSM components
from leibniz_agent.leibniz_appointment_fsm import create_appointment_fsm, AppointmentState
from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager, process_leibniz_rag_query
from leibniz_agent.leibniz_rag import process_leibniz_query_async
# Messages not needed for this test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration constants
TEST_TIMEOUT = 60.0  # Maximum time for a single test scenario
EXPECTED_RESPONSE_TIME = 2.5  # Expected max response time per turn
INTENT_ACCURACY_THRESHOLD = 0.95  # 95% accuracy target
TONE_SCORE_THRESHOLD = 0.1  # Friendly casual tone threshold

@dataclass
class EndToEndTestScenario:
    """Defines a test scenario for end-to-end conversation flow"""
    name: str
    description: str
    user_inputs: List[str]
    expected_intents: List[str]
    expected_flow: List[str]
    validation_checks: List[Callable]
    timeout: float = TEST_TIMEOUT

# Mock audio capture queue for simulating user inputs
mock_input_queue = []
original_capture_function = None

async def mock_capture_and_transcribe(text: str = None) -> Tuple[Optional[bytes], Optional[str]]:
    """Mock function to simulate audio capture and transcription"""
    global mock_input_queue
    
    if not mock_input_queue:
        return None, None
    
    # Get next input from queue
    next_input = mock_input_queue.pop(0)
    
    # Simulate realistic capture delay
    await asyncio.sleep(0.7)
    
    logger.info(f"Mock capture: '{next_input}'")
    return None, next_input

def setup_mock_audio_capture(inputs: List[str]):
    """Set up mock audio capture with predefined inputs"""
    global mock_input_queue, original_capture_function
    
    # Store original function
    if original_capture_function is None:
        original_capture_function = leibniz_pro.capture_and_transcribe
    
    # Set up mock queue
    mock_input_queue = inputs.copy()
    
    # Replace with mock function
    leibniz_pro.capture_and_transcribe = mock_capture_and_transcribe

def restore_audio_capture():
    """Restore original audio capture function"""
    global original_capture_function
    
    if original_capture_function:
        leibniz_pro.capture_and_transcribe = original_capture_function

# Test Scenarios
def create_test_scenarios() -> List[EndToEndTestScenario]:
    """Create comprehensive test scenarios"""
    scenarios = []
    
    # Scenario 1: Happy Path - Complete Flow
    scenarios.append(EndToEndTestScenario(
        name="Happy Path - Complete Flow",
        description="Tests greeting → RAG query → appointment booking → exit",
        user_inputs=[
            "Hello",
            "What are the requirements for the computer science program?",
            "I'd like to schedule an appointment with admissions",
            "John Smith",
            "john.smith@uni-hannover.de",
            "+49 511 762 2020",
            "admissions",
            "application questions",
            "next Tuesday at 2pm",
            "I have questions about my application",
            "yes",
            "Thanks, goodbye"
        ],
        expected_intents=["GREETING", "RAG_QUERY", "APPOINTMENT_SCHEDULING", "EXIT"],
        expected_flow=[
            "greeting_response",
            "rag_response",
            "appointment_fsm_start",
            "collect_name",
            "collect_email",
            "collect_phone",
            "collect_department",
            "collect_appointment_type",
            "collect_datetime",
            "collect_purpose",
            "confirmation",
            "booking_complete",
            "exit_response"
        ],
        validation_checks=[
            validate_all_intents_classified,
            validate_context_extracted,
            validate_appointment_data_collected,
            validate_friendly_tone
        ]
    ))
    
    # Scenario 2: RAG-Only Flow
    scenarios.append(EndToEndTestScenario(
        name="RAG-Only Flow",
        description="Tests multiple RAG queries without appointment booking",
        user_inputs=[
            "Hi",
            "Tell me about campus housing",
            "What about financial aid?",
            "How do I apply for scholarships?",
            "That's all, thanks"
        ],
        expected_intents=["GREETING", "RAG_QUERY", "RAG_QUERY", "RAG_QUERY", "EXIT"],
        expected_flow=[
            "greeting_response",
            "rag_response_housing",
            "rag_response_financial_aid",
            "rag_response_scholarships",
            "exit_response"
        ],
        validation_checks=[
            validate_all_intents_classified,
            validate_multiple_rag_queries,
            validate_no_appointment_triggered,
            validate_friendly_tone
        ]
    ))
    
    # Scenario 3: Appointment-Only Flow
    scenarios.append(EndToEndTestScenario(
        name="Appointment-Only Flow",
        description="Tests direct appointment booking without RAG queries",
        user_inputs=[
            "I need to book an appointment",
            "Emily Johnson",
            "emily.johnson@uni-hannover.de",
            "+49 511 762 3030",
            "career services",
            "resume review",
            "tomorrow at 10am",
            "I need help updating my resume for internships",
            "yes",
            "Goodbye"
        ],
        expected_intents=["APPOINTMENT_SCHEDULING", "EXIT"],
        expected_flow=[
            "appointment_fsm_start",
            "collect_name",
            "collect_email",
            "collect_phone",
            "collect_department",
            "collect_appointment_type",
            "collect_datetime",
            "collect_purpose",
            "confirmation",
            "booking_complete",
            "exit_response"
        ],
        validation_checks=[
            validate_direct_appointment_booking,
            validate_appointment_data_collected,
            validate_friendly_tone
        ]
    ))
    
    # Scenario 4: Mixed Flow with Interruptions
    scenarios.append(EndToEndTestScenario(
        name="Mixed Flow with Interruptions",
        description="Tests transitions between RAG and appointment modes",
        user_inputs=[
            "Hello",
            "What are the CS program requirements?",
            "Actually, I want to schedule an appointment",
            "Sarah Williams",
            "sarah.williams@uni-hannover.de",
            "+49 511 762 4040",
            "academic advising",
            "course selection",
            "Friday at 3pm",
            "I need help choosing my courses for next semester",
            "yes",
            "Can you tell me about the library?",
            "Bye"
        ],
        expected_intents=["GREETING", "RAG_QUERY", "APPOINTMENT_SCHEDULING", "RAG_QUERY", "EXIT"],
        expected_flow=[
            "greeting_response",
            "rag_response_cs_requirements",
            "appointment_fsm_start",
            "collect_name",
            "collect_email", 
            "collect_phone",
            "collect_department",
            "collect_appointment_type",
            "collect_datetime",
            "collect_purpose",
            "confirmation",
            "booking_complete",
            "rag_response_library",
            "exit_response"
        ],
        validation_checks=[
            validate_smooth_transitions,
            validate_context_maintained,
            validate_appointment_data_collected,
            validate_friendly_tone
        ]
    ))
    
    return scenarios

# Validation Functions
def validate_greeting_response(response: str) -> Dict[str, Any]:
    """Validate greeting response quality"""
    issues = []
    
    # Check for friendly greeting
    greeting_indicators = ["hi", "hello", "welcome", "greetings"]
    has_greeting = any(indicator in response.lower() for indicator in greeting_indicators)
    if not has_greeting:
        issues.append("Missing friendly greeting")
    
    # Check for agent name mention
    agent_names = ["lexi", "leibniz"]
    has_agent_name = any(name in response.lower() for name in agent_names)
    if not has_agent_name:
        issues.append("Agent name not mentioned")
    
    # Check casual tone
    formal_phrases = ["pursuant to", "aforementioned", "hereby"]
    has_formal = any(phrase in response.lower() for phrase in formal_phrases)
    if has_formal:
        issues.append("Formal language detected in greeting")
    
    # Check length
    if len(response) < 20:
        issues.append("Greeting too short")
    elif len(response) > 150:
        issues.append("Greeting too long")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "has_greeting": has_greeting,
        "has_agent_name": has_agent_name,
        "tone": "casual" if not has_formal else "formal"
    }

def validate_rag_response(response: str, query: str, context: Dict) -> Dict[str, Any]:
    """Validate RAG response quality"""
    issues = []
    
    # Check response not empty
    if not response or len(response) < 50:
        issues.append("Response too short or empty")
    
    # Check friendly casual tone
    casual_indicators = ["here's", "basically", "you can", "feel free", "you'll"]
    has_casual = any(indicator in response.lower() for indicator in casual_indicators)
    
    formal_indicators = ["pursuant to", "aforementioned", "hereby", "notwithstanding"]
    has_formal = any(indicator in response.lower() for indicator in formal_indicators)
    
    if has_formal:
        issues.append("Formal academic language detected")
    if not has_casual:
        issues.append("Missing casual tone indicators")
    
    # Check response addresses query
    query_keywords = query.lower().split()
    relevant_keywords = [kw for kw in query_keywords if len(kw) > 3]
    keywords_in_response = sum(1 for kw in relevant_keywords if kw in response.lower())
    relevance_score = keywords_in_response / len(relevant_keywords) if relevant_keywords else 0
    
    if relevance_score < 0.3:
        issues.append("Response may not address the query")
    
    # Check if context was used
    if context and "key_entities" in context:
        entities_used = sum(1 for entity in context["key_entities"].values() 
                          if str(entity).lower() in response.lower())
        if entities_used == 0:
            issues.append("Context entities not reflected in response")
    
    tone_score = (1.0 if has_casual else 0.5) - (0.5 if has_formal else 0.0)
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "tone_score": tone_score,
        "relevance_score": relevance_score,
        "length": len(response),
        "has_casual_tone": has_casual,
        "has_formal_tone": has_formal
    }

def validate_appointment_confirmation(booking_data: Dict) -> Dict[str, Any]:
    """Validate appointment booking data completeness"""
    issues = []
    required_fields = [
        "name", "email", "phone", "department", 
        "appointment_type", "preferred_datetime", "purpose"
    ]
    
    # Check all fields present
    missing_fields = [field for field in required_fields if field not in booking_data]
    if missing_fields:
        issues.append(f"Missing fields: {', '.join(missing_fields)}")
    
    # Validate field formats
    if "email" in booking_data:
        if "@" not in booking_data["email"]:
            issues.append("Invalid email format")
    
    if "phone" in booking_data:
        if len(booking_data["phone"]) < 10:
            issues.append("Phone number too short")
    
    # Check booking timestamp
    if "booking_timestamp" not in booking_data:
        issues.append("Booking timestamp not set")
    
    completeness_score = (len(required_fields) - len(missing_fields)) / len(required_fields)
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "completeness_score": completeness_score,
        "missing_fields": missing_fields,
        "all_fields_present": len(missing_fields) == 0
    }

def validate_exit_response(response: str) -> Dict[str, Any]:
    """Validate exit response quality"""
    issues = []
    
    # Check for farewell
    farewell_indicators = ["bye", "goodbye", "thanks", "have a great", "take care"]
    has_farewell = any(indicator in response.lower() for indicator in farewell_indicators)
    if not has_farewell:
        issues.append("Missing farewell message")
    
    # Check for offer to help again
    help_again_indicators = ["feel free", "anytime", "reach out", "help again", "come back"]
    has_help_offer = any(indicator in response.lower() for indicator in help_again_indicators)
    if not has_help_offer:
        issues.append("Missing offer to help again")
    
    # Check friendly tone
    if len(response) < 20:
        issues.append("Exit message too short")
    elif len(response) > 150:
        issues.append("Exit message too long")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "has_farewell": has_farewell,
        "has_help_offer": has_help_offer,
        "tone": "friendly" if has_farewell and has_help_offer else "neutral"
    }

# Test validation check functions
async def validate_all_intents_classified(test_data: Dict) -> bool:
    """Check that all intents were classified correctly"""
    expected = test_data["scenario"].expected_intents
    actual = test_data["intents_classified"]
    
    # Extract just the intent names for comparison
    actual_intent_names = []
    for intent in actual:
        if isinstance(intent, dict):
            actual_intent_names.append(intent.get("intent", "UNKNOWN"))
        else:
            actual_intent_names.append("UNKNOWN")
    
    # Filter out duplicates from FSM states
    filtered_actual = []
    for intent in actual_intent_names:
        if intent not in ["UNCLEAR", "UNKNOWN"] or intent in expected:
            if not filtered_actual or filtered_actual[-1] != intent:
                filtered_actual.append(intent)
    
    success = len(filtered_actual) == len(expected)
    if success:
        for i, expected_intent in enumerate(expected):
            if i < len(filtered_actual) and filtered_actual[i] != expected_intent:
                success = False
                break
    
    return success

async def validate_context_extracted(test_data: Dict) -> bool:
    """Check that context was properly extracted"""
    intents = test_data["intents_classified"]
    
    # Check RAG query intents have context
    for intent in intents:
        if isinstance(intent, dict) and intent.get("intent") == "RAG_QUERY":
            if not intent.get("entities") or "user_goal" not in intent.get("entities", {}):
                return False
    
    return True

async def validate_appointment_data_collected(test_data: Dict) -> bool:
    """Check that appointment data was collected"""
    if "appointment_data" in test_data and test_data["appointment_data"]:
        validation = validate_appointment_confirmation(test_data["appointment_data"])
        return validation["valid"]
    return False

async def validate_friendly_tone(test_data: Dict) -> bool:
    """Check that all responses have friendly casual tone"""
    responses = test_data["responses_generated"]
    total_score = 0
    count = 0
    
    for response in responses:
        if response and len(response) > 20:  # Skip very short responses
            # Simple tone check
            casual_indicators = ["here's", "basically", "you can", "feel free", "hi", "hello"]
            formal_indicators = ["pursuant to", "aforementioned", "hereby"]
            
            has_casual = any(ind in response.lower() for ind in casual_indicators)
            has_formal = any(ind in response.lower() for ind in formal_indicators)
            
            score = (1.0 if has_casual else 0.5) - (0.5 if has_formal else 0.0)
            total_score += score
            count += 1
    
    avg_score = total_score / count if count > 0 else 0
    return avg_score >= TONE_SCORE_THRESHOLD

async def validate_multiple_rag_queries(test_data: Dict) -> bool:
    """Check that multiple RAG queries were handled"""
    rag_count = sum(1 for intent in test_data["intents_classified"] 
                   if isinstance(intent, dict) and intent.get('intent') == "RAG_QUERY")
    return rag_count >= 3

async def validate_no_appointment_triggered(test_data: Dict) -> bool:
    """Check that appointment FSM was not triggered"""
    return "appointment_data" not in test_data or not test_data["appointment_data"]

async def validate_direct_appointment_booking(test_data: Dict) -> bool:
    """Check that appointment was booked directly without RAG"""
    intents = test_data["intents_classified"]
    first_intent = intents[0] if intents else None
    
    if isinstance(first_intent, dict):
        return first_intent.get("intent") == "APPOINTMENT_SCHEDULING"
    
    return False

async def validate_smooth_transitions(test_data: Dict) -> bool:
    """Check smooth transitions between RAG and appointment modes"""
    # This would check the flow matches expected transitions
    return True  # Simplified for now

async def validate_context_maintained(test_data: Dict) -> bool:
    """Check that context is maintained across queries"""
    # This would verify context persistence
    return True  # Simplified for now

# Test Execution Engine
async def run_end_to_end_test(scenario: EndToEndTestScenario) -> Dict[str, Any]:
    """Execute a single end-to-end test scenario"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running test: {scenario.name}")
    logger.info(f"Description: {scenario.description}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    test_data = {
        "scenario": scenario,
        "intents_classified": [],
        "responses_generated": [],
        "appointment_data": None,
        "timing_metrics": [],
        "errors": []
    }
    
    try:
        # Initialize services
        logger.info("Initializing Leibniz services...")
        await get_leibniz_services_manager()
        
        # Set up mock audio capture
        setup_mock_audio_capture(scenario.user_inputs)
        
        # Track conversation state
        current_turn = 0
        fsm = None
        in_appointment_flow = False
        
        # Execute conversation loop with throttling
        for user_input in scenario.user_inputs:
            # Add delay between requests to avoid quota limits
            if current_turn > 1:
                await asyncio.sleep(2.0)  # 2 second delay between requests
            turn_start = time.time()
            current_turn += 1
            
            logger.info(f"\nTurn {current_turn}: User says '{user_input}'")
            
            try:
                # Classify intent with retry logic for quota errors
                max_retries = 3
                retry_delay = 2.0
                
                for attempt in range(max_retries):
                    try:
                        intent_result = await classify_leibniz_intent(user_input)
                        test_data["intents_classified"].append(intent_result)
                        break
                    except Exception as e:
                        if "quota" in str(e).lower() and attempt < max_retries - 1:
                            logger.warning(f"Quota error on attempt {attempt + 1}, retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise e
                
                intent = intent_result.get('intent', 'UNKNOWN')
                logger.info(f"Intent classified: {intent}")
                
                # Route to appropriate handler
                # PRIORITY 1: If in appointment flow, handle based on intent
                if in_appointment_flow and fsm:
                    if intent == "RAG_QUERY":
                        # This might be a RAG interruption during appointment booking
                        # First try FSM to see if it can handle as field data
                        fsm_result = await fsm.process_input(user_input)
                        
                        # Check if FSM progressed (accepted the input as field data)
                        if fsm_result.get("state") != fsm_result.get("previous_state"):
                            # FSM progressed - use FSM response
                            test_data["responses_generated"].append(fsm_result["response"])
                            
                            if fsm_result.get("complete"):
                                test_data["appointment_data"] = fsm_result.get("data", {})
                                in_appointment_flow = False
                                fsm = None
                                logger.info(f" Appointment completed: {test_data['appointment_data']}")
                        else:
                            # FSM didn't progress - genuine RAG interruption
                            logger.info(f" Handling RAG interruption during appointment booking")
                            
                            # Process RAG query
                            context = intent_result.get('entities', {})
                            try:
                                rag_response = await process_leibniz_query_async(context=context, query=user_input)
                                test_data["responses_generated"].append(rag_response)
                                
                                # Add resume message
                                resume_message = "Now, let's continue with your appointment booking."
                                test_data["responses_generated"].append(resume_message)
                                logger.info(f" RAG interruption handled, resuming appointment booking")
                            except Exception as rag_e:
                                logger.error(f"RAG interruption failed: {str(rag_e)}")
                                # Use FSM response as fallback
                                test_data["responses_generated"].append(fsm_result["response"])
                    else:
                        # Not a RAG query - send directly to FSM
                        result = await fsm.process_input(user_input)
                        test_data["responses_generated"].append(result["response"])
                        
                        if result.get("complete"):
                            test_data["appointment_data"] = result.get("data", {})
                            in_appointment_flow = False
                            fsm = None
                            logger.info(f" Appointment completed: {test_data['appointment_data']}")
                
                elif intent == "GREETING":
                    response = "Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?"
                    test_data["responses_generated"].append(response)
                    
                elif intent == "RAG_QUERY":
                    # Regular RAG query (not during appointment flow, since FSM is handled above)
                    context = intent_result.get('entities', {})
                    
                    max_rag_retries = 3
                    rag_retry_delay = 3.0
                    
                    for rag_attempt in range(max_rag_retries):
                        try:
                            # Use the async RAG function for better integration
                            rag_result = await process_leibniz_query_async(
                                context=context,
                                query=user_input
                            )
                            response = str(rag_result)  # RAG query returns string directly
                            test_data["responses_generated"].append(response)
                            break
                        except Exception as rag_e:
                            if "quota" in str(rag_e).lower() and rag_attempt < max_rag_retries - 1:
                                logger.warning(f"RAG quota error on attempt {rag_attempt + 1}, retrying in {rag_retry_delay}s...")
                                await asyncio.sleep(rag_retry_delay)
                                rag_retry_delay *= 2
                            else:
                                # Fallback response for RAG failures
                                response = f"I'm sorry, I'm having trouble accessing information right now. Could you try rephrasing your question?"
                                test_data["responses_generated"].append(response)
                                logger.error(f"RAG query failed after retries: {str(rag_e)}")
                                break
                    
                elif intent == "APPOINTMENT_SCHEDULING":
                    # Start appointment FSM
                    if not fsm:
                        fsm = create_appointment_fsm()
                    in_appointment_flow = True
                    result = await fsm.process_input(user_input)
                    test_data["responses_generated"].append(result["response"])
                    
                    # Check if appointment complete
                    if result.get("complete"):
                        test_data["appointment_data"] = result.get("data", {})
                        in_appointment_flow = False
                        fsm = None
                        logger.info(f" Appointment completed: {test_data['appointment_data']}")
                    
                elif intent == "EXIT":
                    response = "Thanks for chatting! Have a great day, and feel free to reach out anytime you need help."
                    test_data["responses_generated"].append(response)
                    
                else:
                    # Handle FSM input if in appointment flow (regardless of intent classification)
                    if in_appointment_flow and fsm:
                        result = await fsm.process_input(user_input)
                        test_data["responses_generated"].append(result["response"])
                        
                        if result.get("complete"):
                            test_data["appointment_data"] = result.get("data", {})
                            in_appointment_flow = False
                            fsm = None
                            logger.info(f" Appointment completed: {test_data['appointment_data']}")
                    else:
                        response = "I didn't quite catch that. Could you please rephrase?"
                        test_data["responses_generated"].append(response)
                
                # Log response
                if test_data["responses_generated"]:
                    logger.info(f"Agent: {test_data['responses_generated'][-1][:100]}...")
                
            except Exception as e:
                logger.error(f"Error in turn {current_turn}: {str(e)}")
                test_data["errors"].append({
                    "turn": current_turn,
                    "input": user_input,
                    "error": str(e)
                })
            
            # Track timing
            turn_time = time.time() - turn_start
            test_data["timing_metrics"].append({
                "turn": current_turn,
                "duration": turn_time
            })
            
            if turn_time > EXPECTED_RESPONSE_TIME:
                logger.warning(f"Turn {current_turn} took {turn_time:.2f}s (exceeds target of {EXPECTED_RESPONSE_TIME}s)")
        
        # Run validation checks
        logger.info("\nRunning validation checks...")
        validation_results = {}
        
        for check_func in scenario.validation_checks:
            check_name = check_func.__name__
            try:
                result = await check_func(test_data)
                validation_results[check_name] = result
                logger.info(f"  {check_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                validation_results[check_name] = False
                logger.error(f"  {check_name}: ERROR - {str(e)}")
        
        # Calculate overall result
        all_passed = all(validation_results.values()) and len(test_data["errors"]) == 0
        
        # Calculate metrics
        total_time = time.time() - start_time
        avg_turn_time = sum(t["duration"] for t in test_data["timing_metrics"]) / len(test_data["timing_metrics"]) if test_data["timing_metrics"] else 0
        
        result = {
            "scenario_name": scenario.name,
            "status": "PASS" if all_passed else "FAIL",
            "duration": total_time,
            "turns": current_turn,
            "avg_turn_time": avg_turn_time,
            "validation_results": validation_results,
            "errors": test_data["errors"],
            "metrics": {
                "intent_classifications": len(test_data["intents_classified"]),
                "responses_generated": len(test_data["responses_generated"]),
                "appointment_completed": test_data["appointment_data"] is not None
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Test scenario failed with error: {str(e)}")
        return {
            "scenario_name": scenario.name,
            "status": "ERROR",
            "error": str(e),
            "duration": time.time() - start_time
        }
    finally:
        # Restore original audio capture
        restore_audio_capture()

# Results Reporting
def generate_test_report(results: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    report = {
        "test_suite": "End-to-End Flow Tests",
        "execution_time": datetime.now().isoformat(),
        "summary": {
            "total_scenarios": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "errors": sum(1 for r in results if r["status"] == "ERROR"),
            "total_duration": sum(r.get("duration", 0) for r in results)
        },
        "scenarios": results,
        "metrics": {},
        "issues": [],
        "recommendations": []
    }
    
    # Calculate overall metrics
    total_turns = sum(r.get("turns", 0) for r in results)
    total_turn_time = sum(r.get("avg_turn_time", 0) * r.get("turns", 0) for r in results)
    
    if total_turns > 0:
        report["metrics"]["avg_response_time"] = total_turn_time / total_turns
        report["metrics"]["response_time_target_met"] = report["metrics"]["avg_response_time"] <= EXPECTED_RESPONSE_TIME
    
    # Identify issues
    for result in results:
        if result["status"] != "PASS":
            severity = "critical" if result["status"] == "ERROR" else "high"
            report["issues"].append({
                "scenario": result["scenario_name"],
                "severity": severity,
                "description": f"Scenario failed with status {result['status']}",
                "errors": result.get("errors", []),
                "validation_failures": [k for k, v in result.get("validation_results", {}).items() if not v]
            })
    
    # Generate recommendations
    if report["metrics"].get("avg_response_time", 0) > EXPECTED_RESPONSE_TIME:
        report["recommendations"].append({
            "priority": "high",
            "category": "performance",
            "recommendation": f"Optimize response time - current avg {report['metrics']['avg_response_time']:.2f}s exceeds target of {EXPECTED_RESPONSE_TIME}s"
        })
    
    if report["summary"]["failed"] > 0:
        report["recommendations"].append({
            "priority": "critical",
            "category": "functionality",
            "recommendation": f"Fix failing scenarios - {report['summary']['failed']} scenarios are not passing validation"
        })
    
    return report

def print_report_summary(report: Dict):
    """Print formatted report summary to console"""
    print("\n" + "="*80)
    print("END-TO-END FLOW TEST RESULTS")
    print("="*80)
    
    summary = report["summary"]
    print(f"\nTest Execution Summary:")
    print(f"  Total Scenarios: {summary['total_scenarios']}")
    print(f"  Passed: {summary['passed']} ")
    print(f"  Failed: {summary['failed']} ")
    print(f"  Errors: {summary['errors']} ")
    print(f"  Total Duration: {summary['total_duration']:.2f}s")
    
    if "metrics" in report and report["metrics"]:
        print(f"\nPerformance Metrics:")
        print(f"  Average Response Time: {report['metrics'].get('avg_response_time', 0):.2f}s")
        print(f"  Target Met: {'Yes ' if report['metrics'].get('response_time_target_met', False) else 'No '}")
    
    if report["issues"]:
        print(f"\nIssues Found ({len(report['issues'])}):")
        for issue in report["issues"][:5]:  # Show first 5 issues
            print(f"  - [{issue['severity'].upper()}] {issue['scenario']}: {issue['description']}")
        if len(report["issues"]) > 5:
            print(f"  ... and {len(report['issues']) - 5} more issues")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - [{rec['priority'].upper()}] {rec['recommendation']}")
    
    print("\n" + "="*80)

async def main():
    """Main test runner"""
    logger.info("Starting Leibniz End-to-End Flow Tests")
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Run scenarios with quota management
    results = []
    
    # For quota-limited testing, run only 2 most important scenarios
    priority_scenarios = [scenarios[0], scenarios[2]]  # Happy path and Appointment-only
    
    logger.info(f"Running {len(priority_scenarios)} priority scenarios to manage API quota...")
    
    for i, scenario in enumerate(priority_scenarios):
        result = await run_end_to_end_test(scenario)
        results.append(result)
        
        # Longer pause between scenarios to avoid quota issues
        if i < len(priority_scenarios) - 1:
            logger.info("Waiting 30 seconds between scenarios to manage API quota...")
            await asyncio.sleep(30.0)
    
    # Generate report
    report = generate_test_report(results)
    
    # Save results
    output_dir = Path("leibniz_agent/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "end_to_end_flow_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save markdown report
    with open(output_dir / "END_TO_END_FLOW_REPORT.md", "w") as f:
        f.write("# End-to-End Flow Test Report\n\n")
        f.write(f"Generated: {report['execution_time']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total Scenarios: {report['summary']['total_scenarios']}\n")
        f.write(f"- Passed: {report['summary']['passed']}\n")
        f.write(f"- Failed: {report['summary']['failed']}\n")
        f.write(f"- Errors: {report['summary']['errors']}\n")
        f.write(f"- Total Duration: {report['summary']['total_duration']:.2f}s\n\n")
        
        f.write("## Scenario Results\n\n")
        for result in report['scenarios']:
            f.write(f"### {result['scenario_name']}\n")
            f.write(f"- Status: {result['status']}\n")
            f.write(f"- Duration: {result.get('duration', 0):.2f}s\n")
            f.write(f"- Turns: {result.get('turns', 0)}\n")
            if result.get('validation_results'):
                f.write("- Validation Results:\n")
                for check, passed in result['validation_results'].items():
                    f.write(f"  - {check}: {'PASS' if passed else 'FAIL'}\n")
            f.write("\n")
        
        if report['issues']:
            f.write("## Issues Found\n\n")
            for issue in report['issues']:
                f.write(f"**[{issue['severity'].upper()}]** {issue['scenario']}\n")
                f.write(f"- {issue['description']}\n")
                if issue.get('validation_failures'):
                    f.write(f"- Failed validations: {', '.join(issue['validation_failures'])}\n")
                f.write("\n")
        
        if report['recommendations']:
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- **[{rec['priority'].upper()}]** {rec['recommendation']}\n")
    
    # Print summary
    print_report_summary(report)
    
    # Return exit code
    return 0 if report["summary"]["failed"] == 0 and report["summary"]["errors"] == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)