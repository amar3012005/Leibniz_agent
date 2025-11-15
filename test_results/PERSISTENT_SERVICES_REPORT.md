# Persistent Services Test Report

Generated: 2025-10-27T01:22:52.603670

## Summary

- Overall Performance: GOOD
- Initialization Time: 0.00s
- Pre-warming Effectiveness: 0.0%
- Cache Hit Rate: 40.0%
- Performance Targets Met: 100.0%
- Concurrent Handling: Working

## Initialization Results

- Services Ready: No
- Initialization Time: 0.00s

## Pre-warming Results

## Caching Results

### intent_classification
- Hit Rate: 40.0%
- Speedup: 8.8x

## Performance Benchmarks

### intent_fast
- Target: 0.100s
- Average: 0.051s
- P95: 0.051s
- Target Met: Yes

### intent_gemini
- Target: 1.500s
- Average: 0.801s
- P95: 0.801s
- Target Met: Yes

### rag_cold
- Target: 2.000s
- Average: 1.501s
- P95: 1.501s
- Target Met: Yes

### rag_warm
- Target: 1.200s
- Average: 0.901s
- P95: 0.901s
- Target Met: Yes

### rag_cached
- Target: 0.050s
- Average: 0.021s
- P95: 0.021s
- Target Met: Yes

## Issues Found

- Initialization error: 'charmap' codec can't encode character '\U0001f393' in position 0: character maps to <undefined>
- Pre-warming error: 'charmap' codec can't encode character '\U0001f393' in position 0: character maps to <undefined>

## Recommendations

- **[CRITICAL]** Fix service initialization issues - not all services are ready
- **[HIGH]** Improve pre-warming effectiveness - less than 50% of tests show improvement
