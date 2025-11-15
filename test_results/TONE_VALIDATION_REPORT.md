# Friendly Tone Validation Test Report

Generated: 2025-10-27T01:23:11.993745

## Summary

- Overall Assessment: NEEDS_IMPROVEMENT
- Overall Tone Score: 0.05
- RAG Friendly Rate: 0.0%
- FSM Friendly Rate: 11.1%
- Error Friendly Rate: 100.0%
- Tone Consistent: Yes
- Formal Violations: 0

## RAG Response Tone

## FSM Prompt Tone

### INIT
- Description: Initialization prompt
- Tone Score: 0.03
- Personalization: Yes
- Result: FAIL

### COLLECT_NAME
- Description: Name collection response
- Tone Score: 0.00
- Personalization: Yes
- Result: FAIL

### COLLECT_EMAIL
- Description: Email collection response
- Tone Score: 0.06
- Personalization: Yes
- Result: FAIL

### COLLECT_PHONE
- Description: Phone collection response
- Tone Score: 0.02
- Personalization: Yes
- Result: FAIL

### COLLECT_DEPARTMENT
- Description: Department selection response
- Tone Score: 0.06
- Personalization: Yes
- Result: FAIL

### COLLECT_APPOINTMENT_TYPE
- Description: Appointment type response
- Tone Score: 0.09
- Personalization: Yes
- Result: FAIL

### COLLECT_DATETIME
- Description: DateTime collection response
- Tone Score: 0.08
- Personalization: Yes
- Result: FAIL

### COLLECT_PURPOSE
- Description: Purpose collection response
- Tone Score: 0.02
- Personalization: Yes
- Result: FAIL

### CONFIRM
- Description: Confirmation response
- Tone Score: 0.17
- Personalization: Yes
- Result: PASS

## Issues Found

- Tone score 0.06 below threshold 0.1
- Tone score 0.03 below threshold 0.1
- Tone score 0.08 below threshold 0.1
- Tone score 0.02 below threshold 0.1
- Tone score 0.00 below threshold 0.1
- Tone score 0.09 below threshold 0.1
- Personality inconsistency - score range 0.27 too wide
- RAG tone testing error: 'charmap' codec can't encode character '\U0001f393' in position 0: character maps to <undefined>
- No friendly indicators in substantial text

## Recommendations

- **[HIGH]** Improve RAG response tone - only 0.0% friendly
- **[HIGH]** Improve FSM prompt tone - only 11.1% friendly
