# End-to-End Flow Test Report

Generated: 2025-10-27T01:42:10.296759

## Summary

- Total Scenarios: 2
- Passed: 1
- Failed: 1
- Errors: 0
- Total Duration: 60.81s

## Scenario Results

### Happy Path - Complete Flow
- Status: FAIL
- Duration: 37.64s
- Turns: 12
- Validation Results:
  - validate_all_intents_classified: FAIL
  - validate_context_extracted: FAIL
  - validate_appointment_data_collected: PASS
  - validate_friendly_tone: PASS

### Appointment-Only Flow
- Status: PASS
- Duration: 23.17s
- Turns: 10
- Validation Results:
  - validate_direct_appointment_booking: PASS
  - validate_appointment_data_collected: PASS
  - validate_friendly_tone: PASS

## Issues Found

**[HIGH]** Happy Path - Complete Flow
- Scenario failed with status FAIL
- Failed validations: validate_all_intents_classified, validate_context_extracted

## Recommendations

- **[CRITICAL]** Fix failing scenarios - 1 scenarios are not passing validation
