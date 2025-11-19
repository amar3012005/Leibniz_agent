# End-to-End Flow Test Report

Generated: 2025-11-18T00:22:59.185486

## Summary

- Total Scenarios: 2
- Passed: 0
- Failed: 2
- Errors: 0
- Total Duration: 66.75s

## Scenario Results

### Happy Path - Complete Flow
- Status: FAIL
- Duration: 43.13s
- Turns: 12
- Validation Results:
  - validate_all_intents_classified: FAIL
  - validate_context_extracted: FAIL
  - validate_appointment_data_collected: FAIL
  - validate_friendly_tone: PASS

### Appointment-Only Flow
- Status: FAIL
- Duration: 23.62s
- Turns: 10
- Validation Results:
  - validate_direct_appointment_booking: PASS
  - validate_appointment_data_collected: FAIL
  - validate_friendly_tone: PASS

## Issues Found

**[HIGH]** Happy Path - Complete Flow
- Scenario failed with status FAIL
- Failed validations: validate_all_intents_classified, validate_context_extracted, validate_appointment_data_collected

**[HIGH]** Appointment-Only Flow
- Scenario failed with status FAIL
- Failed validations: validate_appointment_data_collected

## Recommendations

- **[CRITICAL]** Fix failing scenarios - 2 scenarios are not passing validation
