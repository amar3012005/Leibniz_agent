# English Pipeline Test Report

Generated: 2025-10-27T01:21:56.931753

## Summary

- Overall Pass Rate: 83.3%
- STT Pass Rate: 100.0%
- TTS Pass Rate: 0.0%
- Round-trip Pass Rate: 50.0%
- Language Detection Accuracy: 100.0%
- Available Providers: 0

## STT Results

### Clear English Speech
- Input: `Hello, welcome to Leibniz University`
- Transcript: `Hello, welcome to Leibniz University`
- Confidence: 0.95
- Language: en-US
- Result: PASS

### University-Specific Terms
- Input: `What are the computer science program admission requirements?`
- Transcript: `What are the computer science program admission requirements?`
- Confidence: 0.95
- Language: en-US
- Result: PASS

### Appointment Booking Request
- Input: `I'd like to schedule an appointment with the admissions office`
- Transcript: `I'd like to schedule an appointment with the admissions office`
- Confidence: 0.95
- Language: en-US
- Result: PASS

### Complex Academic Query
- Input: `Can you tell me about financial aid scholarships for international students?`
- Transcript: `Can you tell me about financial aid scholarships for international students?`
- Confidence: 0.95
- Language: en-US
- Result: PASS

## TTS Results

## Issues Found

- TTS initialization error: 'charmap' codec can't encode characters in position 0-1: character maps to <undefined>
- High WER: 50.00%

## Recommendations

- **[HIGH]** Improve TTS quality - only 0.0% pass rate
- **[MEDIUM]** Configure additional TTS providers for redundancy - only 0 available
- **[HIGH]** Improve round-trip accuracy - only 50.0% pass rate
