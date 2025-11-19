"""
Quick test for Leibniz Intent Parser (local)

Tests the same queries we used for the Docker intent service.
"""

import asyncio
import time
from leibniz_intent_parser import LeibnizIntentParser


async def test_intent_parser():
    """Test local Leibniz intent parser with multiple queries"""
    
    print("=" * 70)
    print(" LEIBNIZ INTENT PARSER - QUICK TEST")
    print("=" * 70)
    
    # Initialize parser
    print("\n⏳ Initializing parser...")
    parser = LeibnizIntentParser()
    print(" Parser initialized\n")
    
    # Test queries (mix of fast route and Gemini fallback)
    test_queries = [
        # Fast route queries (clear patterns)
        "What are the admission requirements?",
        "Schedule an appointment with academic advisor",
        "I need to change my appointment time",
        "Where is the library located?",
        "Can you tell me about computer science program?",
        "Hello",
        "Goodbye, thanks for your help",
        "Book a meeting with admissions office tomorrow",
        
        # Gemini fallback queries (complex/ambiguous - NO question words or university topics)
        "I'm interested in learning more about your institution",
        "Tell me everything please",
        "Need some information",
        "Looking around for options",
        "Just browsing right now",
        "Give me details",
        "I'd like to know more",
        "Provide me with some guidance",
        "Could use some assistance here",
        "Help me out",
    ]
    
    results = []
    
    print("=" * 70)
    print("TESTING INTENT CLASSIFICATION")
    print("=" * 70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 70)
        
        start_time = time.time()
        
        try:
            # Classify intent
            result = await parser.classify_intent(query)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Extract key info
            intent = result.get("intent", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            fast_route = result.get("fast_route", False)
            method = "fast_route" if fast_route else "gemini_llm"
            
            # Get context if available
            context = result.get("context", {})
            user_goal = context.get("user_goal", "N/A")
            key_entities = context.get("key_entities", {})
            extracted_meaning = context.get("extracted_meaning", "N/A")
            
            print(f"   Intent:     {intent}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Method:     {method}")
            print(f"   Time:       {elapsed_ms:.2f}ms")
            
            if user_goal != "N/A":
                print(f"   Goal:       {user_goal}")
            
            if key_entities:
                print(f"   Entities:   {key_entities}")
            
            if extracted_meaning != "N/A":
                print(f"   Meaning:    {extracted_meaning}")
            
            results.append({
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "method": method,
                "time_ms": elapsed_ms,
                "success": True
            })
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"    Error: {e}")
            print(f"   Time:   {elapsed_ms:.2f}ms")
            
            results.append({
                "query": query,
                "error": str(e),
                "time_ms": elapsed_ms,
                "success": False
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful
    
    avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0
    
    fast_route_count = sum(1 for r in results if r.get("method") == "fast_route")
    gemini_count = sum(1 for r in results if r.get("method") == "gemini_llm")
    
    print(f"\nTotal queries:     {len(results)}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Average time:      {avg_time:.2f}ms")
    print(f"Fast route:        {fast_route_count} ({fast_route_count/len(results)*100:.1f}%)")
    print(f"Gemini LLM:        {gemini_count} ({gemini_count/len(results)*100:.1f}%)")
    
    # Intent distribution
    print("\nIntent Distribution:")
    intent_counts = {}
    for r in results:
        if r.get("success", False):
            intent = r.get("intent", "UNKNOWN")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent:25s}: {count}")
    
    # Performance stats from parser
    print("\n" + "=" * 70)
    print("PARSER PERFORMANCE STATS")
    print("=" * 70)
    stats = parser.get_performance_stats()
    print(f"\nTotal classifications: {stats.get('total_classifications', 0)}")
    print(f"Fast route:            {stats.get('fast_route_count', 0)}")
    print(f"Gemini route:          {stats.get('gemini_route_count', 0)}")
    print(f"Cache hits:            {stats.get('cache_hits', 0)}")
    print(f"Average confidence:    {stats.get('average_confidence', 0.0):.2%}")
    
    print("\n Test completed!\n")


if __name__ == "__main__":
    asyncio.run(test_intent_parser())
