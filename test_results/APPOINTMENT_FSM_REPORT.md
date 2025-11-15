# Appointment FSM Test Report

Generated: 2025-11-01T23:00:07.878977

## Summary

- Overall FSM Health: NEEDS_IMPROVEMENT
- Scenario Success Rate: 14.3%
- Validation Success Rate: 65.2%
- Edge Case Success Rate: 100.0%
- Datetime Parsing Rate: 0.0%

## Scenario Results

### Happy Path - All Valid Inputs
- Description: Tests perfect flow with all valid inputs on first try
- Duration: 0.01s
- Steps: 16
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

### Validation Errors with Retries
- Description: Tests retry logic for validation failures
- Duration: 0.01s
- Steps: 19
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

### Natural Language Date/Time Parsing
- Description: Tests various date/time input formats
- Duration: 0.01s
- Steps: 16
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

### Cancellation Mid-Flow
- Description: Tests cancellation at various states
- Duration: 0.00s
- Steps: 4
- Final State: cancelled
- Expected Outcome: CANCELLED
- Result: PASS

### Correction During Confirmation
- Description: Tests field correction during final confirmation
- Duration: 0.01s
- Steps: 20
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

### Field Confirmation with Rejection
- Description: Tests per-field confirmation with rejection and re-entry
- Duration: 0.01s
- Steps: 18
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

### Empty Response Defaults to Yes
- Description: Tests that empty responses default to yes after 2 attempts
- Duration: 0.01s
- Steps: 19
- Final State: collect_appointment_type
- Expected Outcome: COMPLETE
- Result: FAIL

## Edge Case Results

### Empty Input
- Input: `...`
- Expected: Handled gracefully with helpful prompt
- Handled Gracefully: Yes
- Result: PASS

### Whitespace Only
- Input: `   ...`
- Expected: Treated as empty, retry requested
- Handled Gracefully: Yes
- Result: PASS

### Special Characters in Name
- Input: `O'Brien-Smith...`
- Expected: Accepted as valid name
- Handled Gracefully: Yes
- Result: PASS

### International Phone
- Input: `+1 555 123 4567...`
- Expected: Accepted as valid international number
- Handled Gracefully: Yes
- Result: PASS

### University Email
- Input: `student@uni-hannover.de...`
- Expected: Accepted and noted as preferred
- Handled Gracefully: Yes
- Result: PASS

### Non-University Email
- Input: `personal@gmail.com...`
- Expected: Accepted but note university email preference
- Handled Gracefully: Yes
- Result: PASS

### Past Date
- Input: `yesterday...`
- Expected: Rejected with request for future date
- Handled Gracefully: Yes
- Result: PASS

### Far Future Date
- Input: `next year...`
- Expected: Accepted but warn about booking window
- Handled Gracefully: Yes
- Result: PASS

### Ambiguous Department
- Input: `I need help...`
- Expected: Ask for clarification with department list
- Handled Gracefully: Yes
- Result: PASS

### Very Long Purpose
- Input: `I need help with my application because AAAAAAAAAA...`
- Expected: Truncated to 500 characters
- Handled Gracefully: Yes
- Result: PASS

## Issues Found

- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'
- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; 'NoneType' object is not subscriptable; I didn't quite catch that.; I didn't quite catch that.; I didn't quite catch that.; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'
- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'
- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'; 'financial_aid'
- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'; 'admissions'
- Wrong outcome: expected COMPLETE, got collect_appointment_type
- Errors occurred: 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'; 'career_services'
- Invalid name 'Test User' was accepted
- Valid email 'student@gmail.com' was rejected
- Valid email 'test.email@domain.com' was rejected
- Invalid email 'not-email' was accepted
- Invalid phone '123' was accepted
- Invalid phone 'abc' was accepted
- Invalid phone '123-45' was accepted
- Invalid phone '' was accepted
- Failed to parse datetime: 'tomorrow morning'
- Failed to parse datetime: 'next Tuesday at 2pm'
- Failed to parse datetime: 'December 15 at 10:30am'
- Failed to parse datetime: '12/25/2024 14:00'
- Failed to parse datetime: 'next week'
- Failed to parse datetime: 'Friday afternoon'
- Failed to parse datetime: '10am'
- Failed to parse datetime: '3:30 PM'

## Recommendations

- **[HIGH]** Fix scenario failures - only 14.3% success rate
- **[HIGH]** Improve field validation - 65.2% below target 90.0%
- **[MEDIUM]** Improve datetime parsing - only 0.0% formats parsed
