# RAG Cache Integration Guide: Leibniz Agent

## Current Status: TARA vs Leibniz Cache Implementation

### TARA Agent (Production - Has Cache)
**Location**: `rag_cache.py` - Three-layer LRU cache system
**Integration**: `tara_pro_backup.py` lines 652-730 (`handle_rag_query()`)

**Cache Layers**:
- **QueryCache**: 1 hour TTL, 200 max entries for full RAG responses
- **IntentCache**: 30 minutes TTL, 500 max entries for intent classifications  
- **FragmentCache**: 15 minutes TTL, 100 max entries for speculative execution

**Performance Benefits**: 200-600x speedup (1-5ms cache hits vs 500-1500ms full processing)

### Leibniz Agent (Current - No Cache)
**Location**: `leibniz_agent/leibniz_rag.py` - Direct FAISS + Gemini processing
**Status**: No cache integration, every query goes through full RAG pipeline

## Integration Steps for Leibniz

### Step 1: Import Cache Manager
Add to `leibniz_agent/leibniz_rag.py` imports:

```python
from rag_cache import get_rag_cache_manager
```

### Step 2: Initialize Cache Manager
Add to `LeibnizRAG.__init__()`:

```python
# Initialize cache manager (singleton pattern)
self.cache_manager = get_rag_cache_manager(namespace="leibniz")
```

### Step 3: Add Cache Check in process_rag_query()
Modify `process_rag_query()` method - add cache check BEFORE processing:

```python
async def process_rag_query(self, context: Dict[str, Any] = None, query: str = None, streaming_callback: Optional[Callable[[str, bool], None]] = None) -> str:
    # ... existing code ...
    
    # Step 1: Extract query from context (existing logic)
    query_text = ""
    if context:
        query_text = context.get('extracted_meaning', '')
        if not query_text:
            query_text = context.get('user_goal', '')
    if not query_text:
        query_text = query if query else ""
    
    if not query_text:
        return "I'm sorry, I didn't understand your question. Could you please rephrase it?"
    
    # NEW: Check cache FIRST (before any processing)
    cached_result = self.cache_manager.get_query_response(query_text, language='mixed')
    if cached_result:
        logger.info(f"✅ Cache hit for query: '{query_text[:50]}...'")
        return cached_result
    
    # ... existing RAG processing continues ...
```

### Step 4: Add Cache Storage After Processing
Modify the end of `process_rag_query()` - add cache storage AFTER successful generation:

```python
    # Step 10: Humanize response for conversational English (existing)
    final_response = self._humanize_response_english(raw_response, query_text, context, is_first_turn)
    
    # NEW: Cache the successful result
    try:
        cache_metadata = {
            'method': 'leibniz_rag',
            'relevance_score': 0.0,  # Could be enhanced with actual relevance scoring
            'processing_time_ms': timing.get('total_ms', 0)
        }
        self.cache_manager.cache_query_response(query_text, final_response, language='mixed', metadata=cache_metadata)
        logger.debug(f"💾 Cached Leibniz RAG response for query: '{query_text[:50]}...'")
    except Exception as cache_error:
        logger.warning(f"⚠️ Failed to cache response: {cache_error}")
    
    # Step 11: Return response (existing)
    if context and context.get('return_timing'):
        return (final_response, timing)
    else:
        return final_response
```

### Step 5: Configure Cache Namespace
Ensure Leibniz uses separate cache namespace to avoid conflicts with TARA:

```python
# In LeibnizRAG.__init__()
self.cache_manager = get_rag_cache_manager(namespace="leibniz")  # NOT "tara"
```

### Step 6: Handle Cache Errors Gracefully
Wrap cache operations in try-catch to prevent cache failures from breaking RAG:

```python
# Cache check with error handling
try:
    cached_result = self.cache_manager.get_query_response(query_text, language='mixed')
    if cached_result:
        return cached_result
except Exception as cache_error:
    logger.warning(f"⚠️ Cache check failed: {cache_error}")
    # Continue with normal processing
```

## Expected Performance Improvements

### Before Integration (Current Leibniz)
- **Cold queries**: 500-1500ms (full FAISS + Gemini processing)
- **Cache hit rate**: 0% (no cache)
- **Memory usage**: No cache overhead

### After Integration (Cached Leibniz)
- **Cache hits**: 1-5ms (200-600x faster)
- **Cache misses**: 500-1500ms (same as before)
- **Expected hit rate**: 40-60% for common university queries
- **Memory usage**: ~50MB cache files + metadata

## Cache Key Strategy

### Consistent Language Parameter
**CRITICAL**: Always use `language='mixed'` for cache keys:

```python
# ✅ CORRECT
cached = cache_manager.get_query_response(query, language='mixed')
cache_manager.cache_query_response(query, result, language='mixed')

# ❌ WRONG (causes cache misses)
cached = cache_manager.get_query_response(query, language='english')
cache_manager.cache_query_response(query, result, language='english')
```

### Query Normalization
Cache keys use MD5 hash of normalized query text. Ensure queries are consistently formatted before caching.

## Testing Integration

### Basic Cache Test
```python
# Test cache functionality
rag = LeibnizRAG()
query = "What are the admission requirements?"

# First call (cache miss)
result1 = await rag.process_rag_query(query=query)
print(f"First result: {result1[:100]}...")

# Second call (cache hit - should be instant)
result2 = await rag.process_rag_query(query=query)
print(f"Second result: {result2[:100]}...")

# Verify results are identical
assert result1 == result2
print("✅ Cache working correctly")
```

### Performance Benchmark
```python
import time

queries = [
    "What are the library hours?",
    "How do I apply for admission?",
    "What courses are offered?",
    "Where is the registrar's office?"
]

rag = LeibnizRAG()

for query in queries:
    # Warm up (cache miss)
    await rag.process_rag_query(query=query)
    
    # Benchmark (cache hit)
    start = time.time()
    result = await rag.process_rag_query(query=query)
    elapsed = (time.time() - start) * 1000
    print(f"Cache hit: {elapsed:.1f}ms for '{query[:30]}...'")
```

Expected results: <5ms per cache hit vs 500-1500ms for misses.

## Configuration Options

### Cache TTL Settings
Modify in `rag_cache.py` if different TTLs needed for Leibniz:

```python
# Current TARA settings (can be customized for Leibniz)
QUERY_CACHE_TTL = 3600  # 1 hour
INTENT_CACHE_TTL = 1800  # 30 minutes  
FRAGMENT_CACHE_TTL = 900  # 15 minutes
```

### Cache Size Limits
```python
# Current TARA limits (can be adjusted)
QUERY_CACHE_MAX_SIZE = 200
INTENT_CACHE_MAX_SIZE = 500
FRAGMENT_CACHE_MAX_SIZE = 100
```

## Error Handling

### Cache Failures Don't Break RAG
- Cache misses fall back to normal processing
- Cache write failures are logged but don't prevent responses
- Corrupted cache files are automatically rebuilt

### Monitoring Cache Health
```python
# Check cache statistics
stats = rag.cache_manager.get_cache_stats()
print(f"Query cache: {stats['query_cache']['size']}/{stats['query_cache']['max_size']} entries")
print(f"Hit rate: {stats['query_cache']['hit_rate']:.1%}")
```

## Migration Path

### Phase 1: Basic Query Caching (Recommended Start)
1. Add cache manager import and initialization
2. Implement cache check at start of `process_rag_query()`
3. Implement cache storage at end of `process_rag_query()`
4. Test with basic queries

### Phase 2: Intent Caching (Optional)
1. Cache intent classification results in `IntentCache`
2. Implement intent cache checks in intent parser
3. Measure performance improvements

### Phase 3: Fragment Caching (Advanced)
1. Implement speculative execution caching
2. Add fragment-level cache checks
3. Optimize for streaming responses

## Compatibility Notes

### Existing Code Unchanged
- All existing `process_rag_query()` calls work without modification
- Cache is transparent to calling code
- Fallback to uncached behavior if cache fails

### Memory Impact
- Cache files stored in `rag_cache/` directory
- Automatic cleanup of expired entries
- No impact on existing memory usage patterns

## Troubleshooting

### Cache Not Working
1. Check `rag_cache/` directory exists and is writable
2. Verify namespace is set correctly (`"leibniz"`)
3. Check logs for cache-related errors
4. Ensure `language='mixed'` is used consistently

### Performance Issues
1. Monitor cache hit rates (>40% expected)
2. Check cache file sizes (should be <100MB total)
3. Verify MD5 key generation is consistent
4. Profile cache I/O operations if slow

### Data Consistency
1. Cache invalidation happens automatically via TTL
2. Manual cache clearing: `rag.cache_manager.clear_all_caches()`
3. Cache files are JSON - can be inspected manually

This integration will bring Leibniz RAG performance in line with TARA's production caching system, providing significant speed improvements for repeated queries while maintaining full backward compatibility.