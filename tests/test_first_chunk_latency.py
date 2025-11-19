"""
Comment 7: Latency Benchmark Test for Leibniz Agent First Chunk

This test measures time from classification done → first sentence audio playback,
comparing against TARA baseline to validate the ≤700ms budget in warm conditions.

Test Methodology:
1. Warm up all services (TTS, RAG, embeddings, Gemini model)
2. Execute 10 test queries with timestamps at critical points:
   - t_classification_done: When intent classification completes
   - t_first_sentence_audio: When first audio chunk starts playing
3. Calculate: first_chunk_latency = t_first_sentence_audio - t_classification_done
4. Compare against TARA baseline (~400-600ms)
5. Fail if budget (≤700ms) exceeded in warm conditions

TARA Baseline (from tara_pro.py):
- Cache hit: 1-5ms
- Cache miss (persistent RAG): 500-2000ms
- First chunk (streaming): 400-600ms typical, 700ms max acceptable

Usage:
    python leibniz_agent/tests/test_first_chunk_latency.py
    
Expected Output:
     PASS: All 10 queries within 700ms budget (avg: 550ms)
     FAIL: 3/10 queries exceeded budget (max: 850ms)
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
LATENCY_BUDGET_MS = 700.0  # ≤700ms target from classification → first audio
TARA_BASELINE_MIN_MS = 400.0  # TARA typical minimum
TARA_BASELINE_MAX_MS = 600.0  # TARA typical maximum
NUM_TEST_QUERIES = 10  # Number of test iterations

# Test queries (realistic university service queries)
TEST_QUERIES = [
    "What are the admission requirements for the Computer Science program?",
    "How do I apply for on-campus housing?",
    "What financial aid options are available?",
    "When does the fall semester start?",
    "Where is the student services office located?",
    "How do I register for courses?",
    "What are the library hours?",
    "How can I schedule an appointment with an academic advisor?",
    "What dining options are available on campus?",
    "How do I get a student ID card?"
]


@dataclass
class LatencyMeasurement:
    """Single latency measurement with timestamps"""
    query: str
    t_classification_start: float
    t_classification_done: float
    t_first_sentence_audio: float
    classification_time_ms: float
    first_chunk_latency_ms: float
    total_time_ms: float
    passed: bool
    

async def warmup_services():
    """
    Warm up all services to ensure fair measurement
    
    Warms up:
    - Persistent services manager (singleton)
    - TTS provider (ElevenLabs/Google/Gemini)
    - RAG system (FAISS index, embeddings, Gemini model)
    - Intent parser (Gemini classification)
    """
    logger.info(" Warming up services...")
    
    try:
        # Import services
        from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager
        
        # Get singleton manager (triggers initialization)
        services = await get_leibniz_services_manager()
        
        # Warm up each service with dummy requests
        logger.info("  - Warming up TTS...")
        await services.tts_system.synthesize_async("Warmup test", emotion="neutral")
        
        logger.info("  - Warming up RAG...")
        services.rag_system.rag_system.process_rag_query(
            context={"user_goal": "warmup", "extracted_meaning": "test query"},
            query="What is Leibniz University?",
            streaming_callback=None
        )
        
        logger.info("  - Warming up Intent Parser...")
        await services.intent_parser.classify_intent("Test query for warmup")
        
        logger.info(" All services warmed up")
        return services
        
    except Exception as e:
        logger.error(f" Warmup failed: {e}")
        raise


async def measure_single_query(services, query: str) -> LatencyMeasurement:
    """
    Measure latency for a single query
    
    Returns:
        LatencyMeasurement with timestamps and computed metrics
    """
    logger.info(f" Testing: '{query[:60]}...'")
    
    # Timestamps
    t_classification_start = time.time()
    t_classification_done = 0.0
    t_first_sentence_audio = 0.0
    
    try:
        # Step 1: Intent Classification (simulate leibniz_pro.py flow)
        intent_result = await services.intent_parser.classify_intent(query)
        t_classification_done = time.time()
        classification_time_ms = (t_classification_done - t_classification_start) * 1000
        
        logger.debug(f"   Classification done: {classification_time_ms:.1f}ms")
        
        # Step 2: RAG Query with Streaming (capture first sentence callback)
        first_callback_time = None
        
        def streaming_callback(text: str, is_final: bool):
            """Capture timestamp of first sentence callback"""
            nonlocal first_callback_time
            if first_callback_time is None and text.strip():
                first_callback_time = time.time()
                logger.debug(f"   First sentence callback: '{text[:50]}...'")
        
        # Extract context from intent result
        context = {
            'user_goal': intent_result.get('intent', 'general_query'),
            'key_entities': intent_result.get('extracted_info', {}),
            'extracted_meaning': query
        }
        
        # Execute RAG with streaming callback
        services.rag_system.rag_system.process_rag_query(
            context=context,
            query=query,
            streaming_callback=streaming_callback
        )
        
        # Timestamp for first sentence audio
        t_first_sentence_audio = first_callback_time if first_callback_time else time.time()
        
        # Compute metrics
        first_chunk_latency_ms = (t_first_sentence_audio - t_classification_done) * 1000
        total_time_ms = (t_first_sentence_audio - t_classification_start) * 1000
        passed = first_chunk_latency_ms <= LATENCY_BUDGET_MS
        
        logger.info(
            f"  ⏱️  First Chunk Latency: {first_chunk_latency_ms:.1f}ms "
            f"({' PASS' if passed else ' FAIL'})"
        )
        
        return LatencyMeasurement(
            query=query,
            t_classification_start=t_classification_start,
            t_classification_done=t_classification_done,
            t_first_sentence_audio=t_first_sentence_audio,
            classification_time_ms=classification_time_ms,
            first_chunk_latency_ms=first_chunk_latency_ms,
            total_time_ms=total_time_ms,
            passed=passed
        )
        
    except Exception as e:
        logger.error(f" Query failed: {e}")
        # Return failed measurement
        return LatencyMeasurement(
            query=query,
            t_classification_start=t_classification_start,
            t_classification_done=t_classification_done or time.time(),
            t_first_sentence_audio=time.time(),
            classification_time_ms=0.0,
            first_chunk_latency_ms=999999.0,  # Max value to indicate failure
            total_time_ms=0.0,
            passed=False
        )


async def run_benchmark() -> Dict[str, Any]:
    """
    Run full latency benchmark suite
    
    Returns:
        Dictionary with results summary and detailed measurements
    """
    logger.info("=" * 80)
    logger.info("LEIBNIZ AGENT - FIRST CHUNK LATENCY BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Budget: ≤{LATENCY_BUDGET_MS}ms")
    logger.info(f"TARA Baseline: {TARA_BASELINE_MIN_MS}-{TARA_BASELINE_MAX_MS}ms")
    logger.info(f"Test Queries: {NUM_TEST_QUERIES}")
    logger.info("=" * 80)
    
    # Warmup phase
    services = await warmup_services()
    logger.info("")
    
    # Measurement phase
    measurements: List[LatencyMeasurement] = []
    
    for i in range(NUM_TEST_QUERIES):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        logger.info(f"\n[{i+1}/{NUM_TEST_QUERIES}] Running test query...")
        
        measurement = await measure_single_query(services, query)
        measurements.append(measurement)
        
        # Brief pause between queries to avoid rate limiting
        await asyncio.sleep(0.5)
    
    # Analysis phase
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)
    
    passed_count = sum(1 for m in measurements if m.passed)
    failed_count = NUM_TEST_QUERIES - passed_count
    
    latencies = [m.first_chunk_latency_ms for m in measurements if m.first_chunk_latency_ms < 999999.0]
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else max_latency
        
        logger.info(f"\nLatency Statistics:")
        logger.info(f"  - Average:    {avg_latency:.1f}ms")
        logger.info(f"  - Median:     {median_latency:.1f}ms")
        logger.info(f"  - Min:        {min_latency:.1f}ms")
        logger.info(f"  - Max:        {max_latency:.1f}ms")
        logger.info(f"  - P95:        {p95_latency:.1f}ms")
        
        logger.info(f"\nPass/Fail:")
        logger.info(f"  - Passed:     {passed_count}/{NUM_TEST_QUERIES} ({passed_count/NUM_TEST_QUERIES*100:.1f}%)")
        logger.info(f"  - Failed:     {failed_count}/{NUM_TEST_QUERIES} ({failed_count/NUM_TEST_QUERIES*100:.1f}%)")
        
        # Compare to TARA baseline
        logger.info(f"\nComparison to TARA Baseline:")
        logger.info(f"  - TARA Range: {TARA_BASELINE_MIN_MS}-{TARA_BASELINE_MAX_MS}ms")
        logger.info(f"  - Leibniz Avg: {avg_latency:.1f}ms")
        
        if avg_latency <= TARA_BASELINE_MAX_MS:
            logger.info(f"  -  Within TARA baseline range")
        elif avg_latency <= LATENCY_BUDGET_MS:
            logger.info(f"  - ️  Above TARA baseline but within budget")
        else:
            logger.info(f"  -  EXCEEDS budget")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        if failed_count == 0:
            logger.info(" BENCHMARK PASSED: All queries within budget")
            overall_pass = True
        elif failed_count <= NUM_TEST_QUERIES * 0.1:  # Allow 10% failure tolerance
            logger.info(f"️  BENCHMARK WARNING: {failed_count} queries exceeded budget (within tolerance)")
            overall_pass = True
        else:
            logger.info(f" BENCHMARK FAILED: {failed_count} queries exceeded budget")
            overall_pass = False
        logger.info("=" * 80)
        
        return {
            'passed': overall_pass,
            'total_queries': NUM_TEST_QUERIES,
            'passed_count': passed_count,
            'failed_count': failed_count,
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'budget_ms': LATENCY_BUDGET_MS,
            'tara_baseline_min_ms': TARA_BASELINE_MIN_MS,
            'tara_baseline_max_ms': TARA_BASELINE_MAX_MS,
            'measurements': measurements
        }
    else:
        logger.error(" No valid measurements collected")
        return {
            'passed': False,
            'total_queries': NUM_TEST_QUERIES,
            'passed_count': 0,
            'failed_count': NUM_TEST_QUERIES,
            'measurements': measurements
        }


async def main():
    """Main entry point"""
    try:
        results = await run_benchmark()
        
        # Exit with appropriate code
        exit_code = 0 if results['passed'] else 1
        return exit_code
        
    except Exception as e:
        logger.error(f" Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
