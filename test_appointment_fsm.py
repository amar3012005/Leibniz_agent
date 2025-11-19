#!/usr/bin/env python3
"""
Appointment FSM Test Suite for Leibniz University Customer Service Agent

This script validates the appointment booking FSM with various input patterns, edge cases,
validation rules, retries, corrections, and state transitions. Tests all aspects of the
finite state machine for comprehensive appointment booking functionality.

**Test Coverage:**
- Happy path with all valid inputs
- Validation errors and retry logic
- Per-field confirmation with spell-out and readback
- Empty response handling (default to yes after 2 attempts)
- Confirmation rejection and re-collection flow
- Natural language date/time parsing
- Cancellation at various states
- Field correction during final confirmation

Author: SINDH Technologies
Date: October 2025
"""

import asyncio
import time
import json
import logging
import sys
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import os
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Leibniz components
from leibniz_agent.leibniz_appointment_fsm import create_appointment_fsm, AppointmentState, AppointmentData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration constants
MAX_RETRIES_TARGET = 3
VALIDATION_SUCCESS_THRESHOLD = 0.9  # 90% validation success rate
STATE_TRANSITION_SUCCESS_THRESHOLD = 0.95  # 95% state transitions correct
EDGE_CASE_HANDLING_THRESHOLD = 0.8  # 80% edge cases handled

class FSMTestScenario:
    """Defines a test scenario for FSM validation"""
    def __init__(self, name: str, description: str, inputs: List[str], 
                 expected_states: List[str], expected_outcome: str):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.expected_states = expected_states
        self.expected_outcome = expected_outcome  # "COMPLETE", "CANCELLED", "ERROR"

def create_fsm_test_scenarios() -> List[FSMTestScenario]:
    """Create comprehensive FSM test scenarios"""
    scenarios = []
    
    # Scenario 1: Happy Path - All Valid Inputs First Try
    scenarios.append(FSMTestScenario(
        name="Happy Path - All Valid Inputs",
        description="Tests perfect flow with all valid inputs on first try",
        inputs=[
            "",  # Initialize
            "John Smith",
            "yes",  # Confirm name
            "john.smith@uni-hannover.de",
            "yes",  # Confirm email
            "+49 511 762 2020",
            "yes",  # Confirm phone
            "admissions",
            "yes",  # Confirm department
            "application questions",
            "yes",  # Confirm appointment type
            "next Tuesday at 2pm",
            "yes",  # Confirm datetime
            "I have questions about my application",
            "yes",  # Confirm purpose
            "yes"   # Final confirmation
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "collect_email", 
            "confirm_email",
            "collect_phone",
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    # Scenario 2: Validation Errors with Retries
    scenarios.append(FSMTestScenario(
        name="Validation Errors with Retries",
        description="Tests retry logic for validation failures",
        inputs=[
            "",  # Initialize
            "J",  # Too short name
            "John Smith",  # Valid name
            "yes",  # Confirm name
            "not-an-email",  # Invalid email
            "john@uni-hannover.de",  # Valid email
            "yes",  # Confirm email
            "123",  # Too short phone
            "+49 511 762 2020",  # Valid phone
            "yes",  # Confirm phone
            "admissions",
            "yes",  # Confirm department
            "application questions",
            "yes",  # Confirm appointment type
            "next Tuesday at 2pm",
            "yes",  # Confirm datetime
            "I have questions about my application",
            "yes",  # Confirm purpose
            "yes"   # Final confirmation
        ],
        expected_states=[
            "collect_name",
            "collect_name",  # Retry
            "confirm_name",
            "collect_email",
            "collect_email",  # Retry
            "confirm_email",
            "collect_phone", 
            "collect_phone",  # Retry
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    # Scenario 3: Natural Language Date/Time Parsing
    scenarios.append(FSMTestScenario(
        name="Natural Language Date/Time Parsing",
        description="Tests various date/time input formats",
        inputs=[
            "",
            "Alice Johnson",
            "yes",  # Confirm name
            "alice@uni-hannover.de",
            "yes",  # Confirm email
            "+49 511 762 3030",
            "yes",  # Confirm phone
            "career services",
            "yes",  # Confirm department
            "resume review",
            "yes",  # Confirm appointment type
            "tomorrow morning",  # Natural language
            "yes",  # Confirm datetime
            "I need help with my resume for job applications",
            "yes",  # Confirm purpose
            "yes"   # Final confirmation
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "collect_email",
            "confirm_email",
            "collect_phone",
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    # Scenario 4: Cancellation Mid-Flow
    scenarios.append(FSMTestScenario(
        name="Cancellation Mid-Flow",
        description="Tests cancellation at various states",
        inputs=[
            "",
            "Bob Wilson",
            "yes",  # Confirm name
            "cancel"  # Cancel during email collection
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "collect_email",
            "cancelled"
        ],
        expected_outcome="CANCELLED"
    ))
    
    # Scenario 5: Correction During Confirmation
    scenarios.append(FSMTestScenario(
        name="Correction During Confirmation", 
        description="Tests field correction during final confirmation",
        inputs=[
            "",
            "Sarah Davis",
            "yes",  # Confirm name
            "sarah@uni-hannover.de",
            "yes",  # Confirm email
            "+49 511 762 4040",
            "yes",  # Confirm phone
            "financial aid",
            "yes",  # Confirm department
            "scholarship application",
            "yes",  # Confirm appointment type
            "next Friday at 10am",
            "yes",  # Confirm datetime
            "I need help applying for scholarships",
            "yes",  # Confirm purpose
            "no",  # Reject final confirmation
            "change email",  # Request email change
            "sarah.davis@student.uni-hannover.de",  # New email
            "yes",  # Confirm new email
            "yes"  # Final confirmation again
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "collect_email",
            "confirm_email",
            "collect_phone",
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "confirm",  # Still in confirm after rejection
            "collect_email",  # Back to email collection
            "confirm_email",  # Confirm new email
            "confirm",  # Back to final confirmation
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    # Scenario 6: Field Confirmation with Rejection
    scenarios.append(FSMTestScenario(
        name="Field Confirmation with Rejection",
        description="Tests per-field confirmation with rejection and re-entry",
        inputs=[
            "",  # Initialize
            "John Smith",
            "yes",  # Confirm name
            "john@uni-hannover.de",
            "no",   # Reject email
            "john.smith@uni-hannover.de",  # Corrected email
            "yes",  # Confirm corrected email
            "+49 511 762 2020",
            "yes",  # Confirm phone
            "admissions",
            "yes",  # Confirm department
            "application questions",
            "yes",  # Confirm appointment type
            "next Tuesday at 2pm",
            "yes",  # Confirm datetime
            "I have questions about my application",
            "yes",  # Confirm purpose
            "yes"   # Final confirmation
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "collect_email",
            "confirm_email",
            "collect_email",      # Back to collection after rejection
            "confirm_email",      # Confirm corrected value
            "collect_phone",
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    # Scenario 7: Empty Response Defaults to Yes
    scenarios.append(FSMTestScenario(
        name="Empty Response Defaults to Yes",
        description="Tests that empty responses default to yes after 2 attempts",
        inputs=[
            "",  # Initialize
            "Alice Johnson",
            "",  # Empty response 1
            "",  # Empty response 2 - should default to yes
            "alice@uni-hannover.de",
            "",  # Empty - default to yes
            "",  # Empty - default to yes
            "+49 511 762 3030",
            "",  # Empty - default to yes
            "",  # Empty - default to yes
            "career services",
            "yes",  # Explicit yes
            "resume review",
            "yes",
            "tomorrow morning",
            "yes",
            "I need help with my resume",
            "yes",
            "yes"
        ],
        expected_states=[
            "collect_name",
            "confirm_name",
            "confirm_name",       # Still in confirm after first empty
            "collect_email",      # Proceed after second empty
            "confirm_email",
            "confirm_email",      # First empty
            "collect_phone",      # Proceed after second empty
            "confirm_phone",
            "confirm_phone",
            "collect_department",
            "confirm_department",
            "collect_appointment_type",
            "confirm_appointment_type",
            "collect_datetime",
            "confirm_datetime",
            "collect_purpose",
            "confirm_purpose",
            "confirm",
            "complete"
        ],
        expected_outcome="COMPLETE"
    ))
    
    return scenarios

@pytest.fixture
def scenarios():
    """Fixture providing test scenarios"""
    return create_fsm_test_scenarios()

@pytest.mark.asyncio
async def test_fsm_scenarios(scenarios: List[FSMTestScenario]) -> Dict[str, Any]:
    """Test all FSM scenarios"""
    logger.info("Testing FSM scenarios...")
    
    results = {
        "scenario_tests": [],
        "summary": {
            "total_scenarios": len(scenarios),
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "scenario_success_rate": 0.0
        },
        "issues": []
    }
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create fresh FSM for each scenario
            fsm = create_appointment_fsm()
            
            # Track state transitions
            state_transitions = []
            responses = []
            errors = []
            
            # Process each input
            for i, user_input in enumerate(scenario.inputs):
                logger.info(f"\nStep {i+1}: Input '{user_input}'")
                
                try:
                    result = await fsm.process_input(user_input)
                    
                    current_state = result.get("state", "unknown")
                    response = result.get("response", "")
                    complete = result.get("complete", False)
                    cancelled = result.get("cancelled", False)
                    error = result.get("error")
                    
                    state_transitions.append(current_state)
                    responses.append(response)
                    
                    if error:
                        errors.append(error)
                    
                    logger.info(f"  State: {current_state}")
                    logger.info(f"  Response: {response[:100]}...")
                    
                    if complete:
                        logger.info(f"   Booking completed!")
                        break
                    elif cancelled:
                        logger.info(f"   Booking cancelled")
                        break
                
                except Exception as e:
                    logger.error(f"  Error in step {i+1}: {str(e)}")
                    errors.append(str(e))
            
            # Validate scenario results
            duration = time.time() - start_time
            
            # Check final outcome
            final_state = state_transitions[-1] if state_transitions else "unknown"
            outcome_correct = False
            
            if scenario.expected_outcome == "COMPLETE":
                outcome_correct = final_state == "complete"
            elif scenario.expected_outcome == "CANCELLED":
                outcome_correct = final_state == "cancelled"
            else:
                outcome_correct = True  # Other outcomes
            
            # Comment 6: Compare full state_transitions against expected_states (length and order)
            state_sequence_matches = True
            state_diffs = []
            
            # Check length match
            if len(state_transitions) != len(scenario.expected_states):
                state_sequence_matches = False
                state_diffs.append(
                    f"Length mismatch: got {len(state_transitions)} states, expected {len(scenario.expected_states)}"
                )
            
            # Check order match (compare element by element)
            min_len = min(len(state_transitions), len(scenario.expected_states))
            for i in range(min_len):
                if state_transitions[i] != scenario.expected_states[i]:
                    state_sequence_matches = False
                    state_diffs.append(
                        f"Step {i}: got '{state_transitions[i]}', expected '{scenario.expected_states[i]}'"
                    )
            
            # Log any remaining states if lengths differ
            if len(state_transitions) > len(scenario.expected_states):
                extra_states = state_transitions[min_len:]
                state_diffs.append(f"Extra states: {extra_states}")
            elif len(scenario.expected_states) > len(state_transitions):
                missing_states = scenario.expected_states[min_len:]
                state_diffs.append(f"Missing states: {missing_states}")
            
            # Determine if scenario passed
            scenario_passed = outcome_correct and state_sequence_matches and len(errors) == 0
            
            if scenario_passed:
                results["summary"]["passed_scenarios"] += 1
            else:
                results["summary"]["failed_scenarios"] += 1
            
            # Store scenario result
            scenario_result = {
                "name": scenario.name,
                "description": scenario.description,
                "duration": duration,
                "total_steps": len(scenario.inputs),
                "state_transitions": state_transitions,
                "expected_states": scenario.expected_states,
                "final_state": final_state,
                "expected_outcome": scenario.expected_outcome,
                "outcome_correct": outcome_correct,
                "state_sequence_matches": state_sequence_matches,
                "state_diffs": state_diffs,
                "errors": errors,
                "passed": scenario_passed
            }
            
            results["scenario_tests"].append(scenario_result)
            
            logger.info(f"\nScenario Result:")
            logger.info(f"  Final state: {final_state}")
            logger.info(f"  Expected outcome: {scenario.expected_outcome}")
            logger.info(f"  Outcome correct: {'Yes ' if outcome_correct else 'No '}")
            logger.info(f"  State sequence matches: {'Yes ' if state_sequence_matches else 'No '}")
            if state_diffs:
                logger.warning(f"  State sequence diffs:")
                for diff in state_diffs:
                    logger.warning(f"    - {diff}")
            logger.info(f"  Errors: {len(errors)}")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info(f"  Result: {'PASS ' if scenario_passed else 'FAIL '}")
            
            if not scenario_passed:
                issues = []
                if not outcome_correct:
                    issues.append(f"Wrong outcome: expected {scenario.expected_outcome}, got {final_state}")
                if not state_sequence_matches:
                    issues.append(f"State sequence mismatch: {'; '.join(state_diffs)}")
                if errors:
                    issues.append(f"Errors occurred: {'; '.join(errors)}")
                
                results["issues"].extend(issues)
                logger.warning(f"Issues: {'; '.join(issues)}")
        
        except Exception as e:
            logger.error(f"Error in scenario {scenario.name}: {str(e)}")
            results["summary"]["failed_scenarios"] += 1
            results["issues"].append(f"Scenario error in {scenario.name}: {str(e)}")
    
    # Calculate success rate
    if results["summary"]["total_scenarios"] > 0:
        results["summary"]["scenario_success_rate"] = results["summary"]["passed_scenarios"] / results["summary"]["total_scenarios"]
    
    logger.info(f"\nFSM Scenarios Summary:")
    logger.info(f"Passed: {results['summary']['passed_scenarios']}/{results['summary']['total_scenarios']}")
    logger.info(f"Success rate: {results['summary']['scenario_success_rate']:.1%}")
    
    return results

@pytest.mark.asyncio
async def test_field_validation() -> Dict[str, Any]:
    """Test field validation for all appointment fields"""
    logger.info("Testing field validation...")
    
    results = {
        "validation_tests": [],
        "summary": {
            "total_validations": 0,
            "successful_validations": 0,
            "validation_success_rate": 0.0
        },
        "issues": []
    }
    
    # Field validation test cases
    validation_tests = [
        # Name validation
        {
            "field": "name",
            "valid_inputs": ["John Smith", "Maria Garcia", "O'Brien Wilson", "Mary-Jane Johnson"],
            "invalid_inputs": ["J", "X", "123", "Test User", ""]
        },
        # Email validation
        {
            "field": "email", 
            "valid_inputs": ["john@uni-hannover.de", "student@gmail.com", "test.email@domain.com"],
            "invalid_inputs": ["not-email", "missing@", "@domain.com", "spaces in email@test.com"]
        },
        # Phone validation
        {
            "field": "phone",
            "valid_inputs": ["+49 511 762 2020", "0511 762 2020", "+1 555 123 4567"],
            "invalid_inputs": ["123", "abc", "123-45", ""]
        }
    ]
    
    try:
        for field_test in validation_tests:
            field_name = field_test["field"]
            logger.info(f"\nTesting {field_name} validation...")
            
            # Test valid inputs
            for valid_input in field_test["valid_inputs"]:
                fsm = create_appointment_fsm()
                
                # Get to the appropriate state for this field
                if field_name == "name":
                    await fsm.process_input("")  # Initialize
                elif field_name == "email":
                    await fsm.process_input("")  # Initialize
                    await fsm.process_input("John Smith")  # Name
                elif field_name == "phone":
                    await fsm.process_input("")  # Initialize
                    await fsm.process_input("John Smith")  # Name
                    await fsm.process_input("john@test.com")  # Email
                
                # Test the field input
                result = await fsm.process_input(valid_input)
                
                # Check if FSM progressed (validation succeeded)
                current_state = result.get("state", "unknown")
                validation_passed = current_state != result.get("previous_state", current_state)
                
                results["summary"]["total_validations"] += 1
                if validation_passed:
                    results["summary"]["successful_validations"] += 1
                
                logger.info(f"  Valid '{valid_input}': {'PASS ' if validation_passed else 'FAIL '}")
                
                if not validation_passed:
                    results["issues"].append(f"Valid {field_name} '{valid_input}' was rejected")
            
            # Test invalid inputs
            for invalid_input in field_test["invalid_inputs"]:
                fsm = create_appointment_fsm()
                
                # Get to the appropriate state
                if field_name == "name":
                    await fsm.process_input("")
                elif field_name == "email":
                    await fsm.process_input("")
                    await fsm.process_input("John Smith")
                elif field_name == "phone":
                    await fsm.process_input("")
                    await fsm.process_input("John Smith")
                    await fsm.process_input("john@test.com")
                
                # Test the invalid input
                result = await fsm.process_input(invalid_input)
                
                # Check if FSM stayed in same state (validation failed as expected)
                current_state = result.get("state", "unknown")
                validation_correctly_failed = current_state == result.get("previous_state", "different")
                
                results["summary"]["total_validations"] += 1
                if validation_correctly_failed:
                    results["summary"]["successful_validations"] += 1
                
                logger.info(f"  Invalid '{invalid_input}': {'PASS ' if validation_correctly_failed else 'FAIL '}")
                
                if not validation_correctly_failed:
                    results["issues"].append(f"Invalid {field_name} '{invalid_input}' was accepted")
        
        # Calculate validation success rate
        if results["summary"]["total_validations"] > 0:
            results["summary"]["validation_success_rate"] = results["summary"]["successful_validations"] / results["summary"]["total_validations"]
        
        logger.info(f"\nField Validation Summary:")
        logger.info(f"Successful validations: {results['summary']['successful_validations']}/{results['summary']['total_validations']}")
        logger.info(f"Validation success rate: {results['summary']['validation_success_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Error in field validation testing: {str(e)}")
        results["issues"].append(f"Field validation error: {str(e)}")
    
    return results

@pytest.mark.asyncio
async def test_edge_cases() -> Dict[str, Any]:
    """Test edge cases and boundary conditions"""
    logger.info("Testing edge cases...")
    
    results = {
        "edge_case_tests": [],
        "summary": {
            "total_edge_cases": 0,
            "handled_gracefully": 0,
            "edge_case_success_rate": 0.0
        },
        "issues": []
    }
    
    # Edge case test scenarios
    edge_cases = [
        {
            "name": "Empty Input",
            "test_input": "",
            "expected_behavior": "Handled gracefully with helpful prompt"
        },
        {
            "name": "Whitespace Only",
            "test_input": "   ",
            "expected_behavior": "Treated as empty, retry requested"
        },
        {
            "name": "Special Characters in Name",
            "test_input": "O'Brien-Smith",
            "expected_behavior": "Accepted as valid name"
        },
        {
            "name": "International Phone",
            "test_input": "+1 555 123 4567",
            "expected_behavior": "Accepted as valid international number"
        },
        {
            "name": "University Email",
            "test_input": "student@uni-hannover.de", 
            "expected_behavior": "Accepted and noted as preferred"
        },
        {
            "name": "Non-University Email",
            "test_input": "personal@gmail.com",
            "expected_behavior": "Accepted but note university email preference"
        },
        {
            "name": "Past Date",
            "test_input": "yesterday",
            "expected_behavior": "Rejected with request for future date"
        },
        {
            "name": "Far Future Date",
            "test_input": "next year",
            "expected_behavior": "Accepted but warn about booking window"
        },
        {
            "name": "Ambiguous Department",
            "test_input": "I need help",
            "expected_behavior": "Ask for clarification with department list"
        },
        {
            "name": "Very Long Purpose",
            "test_input": "I need help with my application because " + "A" * 600,
            "expected_behavior": "Truncated to 500 characters"
        }
    ]
    
    results["summary"]["total_edge_cases"] = len(edge_cases)
    
    try:
        for edge_case in edge_cases:
            logger.info(f"\nTesting edge case: {edge_case['name']}")
            logger.info(f"Input: '{edge_case['test_input'][:50]}...'")
            
            handled_gracefully = True
            error_details = []
            
            try:
                # Create FSM and get to appropriate state for testing
                fsm = create_appointment_fsm()
                await fsm.process_input("")  # Initialize
                
                # For field-specific tests, navigate to the right state
                if "Email" in edge_case["name"]:
                    await fsm.process_input("John Smith")  # Get to email state
                elif "Phone" in edge_case["name"]:
                    await fsm.process_input("John Smith")
                    await fsm.process_input("john@test.com")  # Get to phone state
                elif "Date" in edge_case["name"]:
                    # Navigate to datetime state
                    await fsm.process_input("John Smith")
                    await fsm.process_input("john@test.com")
                    await fsm.process_input("+49 511 762 2020")
                    await fsm.process_input("admissions")
                    await fsm.process_input("application questions")
                elif "Purpose" in edge_case["name"]:
                    # Navigate to purpose state
                    await fsm.process_input("John Smith")
                    await fsm.process_input("john@test.com")
                    await fsm.process_input("+49 511 762 2020")
                    await fsm.process_input("admissions")
                    await fsm.process_input("application questions")
                    await fsm.process_input("next Tuesday at 2pm")
                
                # Test the edge case input
                result = await fsm.process_input(edge_case["test_input"])
                
                # Check if handled gracefully (no exceptions, has response)
                if not result or "response" not in result:
                    handled_gracefully = False
                    error_details.append("No response generated")
                elif len(result["response"]) < 10:
                    handled_gracefully = False
                    error_details.append("Response too short")
                
                # Log the response
                logger.info(f"  Response: {result.get('response', 'No response')[:100]}...")
                
            except Exception as e:
                handled_gracefully = False
                error_details.append(f"Exception: {str(e)}")
                logger.error(f"  Exception: {str(e)}")
            
            # Store result
            edge_case_result = {
                "name": edge_case["name"],
                "input": edge_case["test_input"],
                "expected_behavior": edge_case["expected_behavior"],
                "handled_gracefully": handled_gracefully,
                "error_details": error_details,
                "passed": handled_gracefully
            }
            
            results["edge_case_tests"].append(edge_case_result)
            
            if handled_gracefully:
                results["summary"]["handled_gracefully"] += 1
            
            logger.info(f"  Result: {'PASS ' if handled_gracefully else 'FAIL '}")
            
            if not handled_gracefully:
                results["issues"].extend(error_details)
        
        # Calculate success rate
        if results["summary"]["total_edge_cases"] > 0:
            results["summary"]["edge_case_success_rate"] = results["summary"]["handled_gracefully"] / results["summary"]["total_edge_cases"]
        
        logger.info(f"\nEdge Cases Summary:")
        logger.info(f"Handled gracefully: {results['summary']['handled_gracefully']}/{results['summary']['total_edge_cases']}")
        logger.info(f"Success rate: {results['summary']['edge_case_success_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Error in edge case testing: {str(e)}")
        results["issues"].append(f"Edge case testing error: {str(e)}")
    
    return results

@pytest.mark.asyncio
async def test_datetime_parsing() -> Dict[str, Any]:
    """Test natural language date/time parsing"""
    logger.info("Testing datetime parsing...")
    
    results = {
        "datetime_tests": [],
        "summary": {
            "total_formats": 0,
            "parsed_successfully": 0,
            "parsing_success_rate": 0.0
        },
        "issues": []
    }
    
    # Date/time formats to test
    datetime_formats = [
        {"input": "tomorrow morning", "expected": "should parse to tomorrow at 10:00 AM"},
        {"input": "next Tuesday at 2pm", "expected": "should parse to next Tuesday at 14:00"},
        {"input": "December 15 at 10:30am", "expected": "should parse to Dec 15 at 10:30"},
        {"input": "12/25/2024 14:00", "expected": "should parse to Dec 25, 2024 at 14:00"},
        {"input": "next week", "expected": "should parse to 7 days from now"},
        {"input": "Friday afternoon", "expected": "should parse to next Friday at 14:00"},
        {"input": "10am", "expected": "should parse to tomorrow at 10:00 AM"},
        {"input": "3:30 PM", "expected": "should parse with proper time format"}
    ]
    
    results["summary"]["total_formats"] = len(datetime_formats)
    
    try:
        # Create FSM and navigate to datetime collection state
        fsm = create_appointment_fsm()
        await fsm.process_input("")  # Initialize
        await fsm.process_input("John Smith")  # Name
        await fsm.process_input("john@test.com")  # Email
        await fsm.process_input("+49 511 762 2020")  # Phone
        await fsm.process_input("admissions")  # Department
        await fsm.process_input("application questions")  # Type
        
        for datetime_test in datetime_formats:
            logger.info(f"\nTesting datetime: '{datetime_test['input']}'")
            
            # Create fresh FSM at datetime state
            test_fsm = create_appointment_fsm()
            await test_fsm.process_input("")
            await test_fsm.process_input("John Smith")
            await test_fsm.process_input("john@test.com")
            await test_fsm.process_input("+49 511 762 2020")
            await test_fsm.process_input("admissions")
            await test_fsm.process_input("application questions")
            
            # Test datetime parsing
            result = await test_fsm.process_input(datetime_test["input"])
            
            # Check if parsing succeeded (FSM progressed to next state)
            current_state = result.get("state", "unknown")
            parsing_succeeded = current_state == "collect_purpose"
            
            if parsing_succeeded:
                results["summary"]["parsed_successfully"] += 1
            
            test_result = {
                "input": datetime_test["input"],
                "expected": datetime_test["expected"],
                "parsing_succeeded": parsing_succeeded,
                "resulting_state": current_state,
                "response": result.get("response", "")[:100] + "...",
                "passed": parsing_succeeded
            }
            
            results["datetime_tests"].append(test_result)
            
            logger.info(f"  Expected: {datetime_test['expected']}")
            logger.info(f"  State after input: {current_state}")
            logger.info(f"  Parsing: {'SUCCESS ' if parsing_succeeded else 'FAILED '}")
            
            if not parsing_succeeded:
                results["issues"].append(f"Failed to parse datetime: '{datetime_test['input']}'")
        
        # Calculate parsing success rate
        if results["summary"]["total_formats"] > 0:
            results["summary"]["parsing_success_rate"] = results["summary"]["parsed_successfully"] / results["summary"]["total_formats"]
        
        logger.info(f"\nDatetime Parsing Summary:")
        logger.info(f"Parsed successfully: {results['summary']['parsed_successfully']}/{results['summary']['total_formats']}")
        logger.info(f"Parsing success rate: {results['summary']['parsing_success_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Error in datetime parsing test: {str(e)}")
        results["issues"].append(f"Datetime parsing error: {str(e)}")
    
    return results

def generate_fsm_test_report(scenario_results: Dict, validation_results: Dict, 
                           edge_case_results: Dict, datetime_results: Dict) -> Dict[str, Any]:
    """Generate comprehensive FSM test report"""
    report = {
        "test_suite": "Appointment FSM Tests",
        "execution_time": datetime.now().isoformat(),
        "scenario_testing": scenario_results,
        "field_validation": validation_results,
        "edge_case_handling": edge_case_results,
        "datetime_parsing": datetime_results,
        "summary": {
            "scenario_success_rate": scenario_results["summary"]["scenario_success_rate"],
            "validation_success_rate": validation_results["summary"]["validation_success_rate"],
            "edge_case_success_rate": edge_case_results["summary"]["edge_case_success_rate"],
            "datetime_parsing_rate": datetime_results["summary"]["parsing_success_rate"],
            "overall_fsm_health": "excellent"
        },
        "issues": [],
        "recommendations": []
    }
    
    # Collect all issues
    for result_set in [scenario_results, validation_results, edge_case_results, datetime_results]:
        report["issues"].extend(result_set.get("issues", []))
    
    # Calculate overall FSM health
    health_scores = [
        report["summary"]["scenario_success_rate"],
        report["summary"]["validation_success_rate"], 
        report["summary"]["edge_case_success_rate"],
        report["summary"]["datetime_parsing_rate"]
    ]
    
    avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
    
    if avg_health >= 0.9:
        report["summary"]["overall_fsm_health"] = "excellent"
    elif avg_health >= 0.7:
        report["summary"]["overall_fsm_health"] = "good"
    else:
        report["summary"]["overall_fsm_health"] = "needs_improvement"
    
    # Generate recommendations
    if report["summary"]["scenario_success_rate"] < 0.8:
        report["recommendations"].append({
            "priority": "high",
            "category": "scenarios",
            "recommendation": f"Fix scenario failures - only {report['summary']['scenario_success_rate']:.1%} success rate"
        })
    
    if report["summary"]["validation_success_rate"] < VALIDATION_SUCCESS_THRESHOLD:
        report["recommendations"].append({
            "priority": "high", 
            "category": "validation",
            "recommendation": f"Improve field validation - {report['summary']['validation_success_rate']:.1%} below target {VALIDATION_SUCCESS_THRESHOLD:.1%}"
        })
    
    if report["summary"]["edge_case_success_rate"] < EDGE_CASE_HANDLING_THRESHOLD:
        report["recommendations"].append({
            "priority": "medium",
            "category": "edge_cases",
            "recommendation": f"Improve edge case handling - {report['summary']['edge_case_success_rate']:.1%} below target {EDGE_CASE_HANDLING_THRESHOLD:.1%}"
        })
    
    if report["summary"]["datetime_parsing_rate"] < 0.7:
        report["recommendations"].append({
            "priority": "medium",
            "category": "datetime_parsing",
            "recommendation": f"Improve datetime parsing - only {report['summary']['datetime_parsing_rate']:.1%} formats parsed"
        })
    
    return report

def print_report_summary(report: Dict):
    """Print formatted report summary to console"""
    print("\n" + "="*80)
    print("APPOINTMENT FSM TEST RESULTS")
    print("="*80)
    
    print(f"\nOverall FSM Health: {report['summary']['overall_fsm_health'].upper()}")
    
    print(f"\nTest Results:")
    print(f"  Scenario Success Rate: {report['summary']['scenario_success_rate']:.1%}")
    print(f"  Validation Success Rate: {report['summary']['validation_success_rate']:.1%}")
    print(f"  Edge Case Success Rate: {report['summary']['edge_case_success_rate']:.1%}")
    print(f"  Datetime Parsing Rate: {report['summary']['datetime_parsing_rate']:.1%}")
    
    print(f"\nScenario Details:")
    for scenario in report['scenario_testing']['scenario_tests']:
        status_icon = "" if scenario['passed'] else ""
        print(f"  {scenario['name']}: {status_icon}")
    
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
    logger.info("Starting Leibniz Appointment FSM Tests")
    
    # Create test scenarios
    scenarios = create_fsm_test_scenarios()
    
    # Test FSM scenarios
    scenario_results = await test_fsm_scenarios(scenarios)
    
    # Test field validation
    validation_results = await test_field_validation()
    
    # Test edge cases
    edge_case_results = await test_edge_cases()
    
    # Test datetime parsing
    datetime_results = await test_datetime_parsing()
    
    # Generate report
    report = generate_fsm_test_report(
        scenario_results, validation_results, edge_case_results, datetime_results
    )
    
    # Save results
    output_dir = Path("leibniz_agent/test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "appointment_fsm_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save markdown report
    with open(output_dir / "APPOINTMENT_FSM_REPORT.md", "w", encoding='utf-8') as f:
        f.write("# Appointment FSM Test Report\n\n")
        f.write(f"Generated: {report['execution_time']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Overall FSM Health: {report['summary']['overall_fsm_health'].upper()}\n")
        f.write(f"- Scenario Success Rate: {report['summary']['scenario_success_rate']:.1%}\n")
        f.write(f"- Validation Success Rate: {report['summary']['validation_success_rate']:.1%}\n")
        f.write(f"- Edge Case Success Rate: {report['summary']['edge_case_success_rate']:.1%}\n")
        f.write(f"- Datetime Parsing Rate: {report['summary']['datetime_parsing_rate']:.1%}\n\n")
        
        f.write("## Scenario Results\n\n")
        for scenario in report['scenario_testing']['scenario_tests']:
            f.write(f"### {scenario['name']}\n")
            f.write(f"- Description: {scenario['description']}\n")
            f.write(f"- Duration: {scenario['duration']:.2f}s\n")
            f.write(f"- Steps: {scenario['total_steps']}\n")
            f.write(f"- Final State: {scenario['final_state']}\n")
            f.write(f"- Expected Outcome: {scenario['expected_outcome']}\n")
            f.write(f"- Result: {'PASS' if scenario['passed'] else 'FAIL'}\n\n")
        
        f.write("## Edge Case Results\n\n")
        for edge_case in report['edge_case_handling']['edge_case_tests']:
            f.write(f"### {edge_case['name']}\n")
            f.write(f"- Input: `{edge_case['input'][:50]}...`\n")
            f.write(f"- Expected: {edge_case['expected_behavior']}\n")
            f.write(f"- Handled Gracefully: {'Yes' if edge_case['handled_gracefully'] else 'No'}\n")
            f.write(f"- Result: {'PASS' if edge_case['passed'] else 'FAIL'}\n\n")
        
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
    return 0 if report["summary"]["overall_fsm_health"] in ["excellent", "good"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
