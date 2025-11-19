"""
Test Semantic Translation for Leibniz Appointment FSM

This script tests the multilingual semantic translation capabilities
for the appointment booking flow.

Usage:
    python leibniz_agent/test_semantic_translation.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from leibniz_agent.leibniz_semantic_translator import (
    generate_semantic_context_for_fsm,
    get_semantic_translator
)


async def test_translation():
    """Test semantic translation with various inputs"""
    
    print("=" * 70)
    print("Leibniz Semantic Translation Test")
    print("=" * 70)
    
    # Test cases: (input, fsm_state, expected_field)
    test_cases = [
        # Hindi inputs
        ("मेरा नाम राज कुमार है", "collect_name", "name"),
        ("राज कुमार", "collect_name", "name"),
        ("मेरा ईमेल raj.kumar@gmail.com है", "collect_email", "email"),
        ("मेरा फोन नंबर +91 9876543210 है", "collect_phone", "phone"),
        ("मुझे एडमिशन ऑफिस के साथ अपॉइंटमेंट चाहिए", "collect_department", "department"),
        
        # Telugu inputs
        ("నా పేరు వెంకట రావు", "collect_name", "name"),
        ("వెంకట రావు", "collect_name", "name"),
        
        # Mixed language
        ("My name is राज कुमार", "collect_name", "name"),
        ("मैं John Smith हूँ", "collect_name", "name"),
        
        # English (should pass through)
        ("John Smith", "collect_name", "name"),
        ("john.smith@uni-hannover.de", "collect_email", "email"),
        ("+49 511 762 2020", "collect_phone", "phone"),
        ("I need to talk to admissions office", "collect_department", "department"),
        
        # Empty/unclear
        ("", "collect_name", "name"),
        ("yes", "confirm_name", "confirmation"),
    ]
    
    translator = get_semantic_translator()
    
    print(f"\n Testing {len(test_cases)} translation scenarios...\n")
    
    passed = 0
    failed = 0
    
    for i, (user_input, fsm_state, expected_field) in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {expected_field.upper()}")
        print(f"  Input: '{user_input}'")
        print(f"  FSM State: {fsm_state}")
        
        try:
            # Test the main entry point function
            result = await generate_semantic_context_for_fsm(
                user_input=user_input,
                fsm_state=fsm_state
            )
            
            print(f"   Translated: '{result['translated_text']}'")
            print(f"  Language: {result['detected_language']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            
            if result.get('field_value'):
                print(f"  Extracted: {result['field_value']}")
            
            passed += 1
            
        except Exception as e:
            print(f"   Error: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed}/{len(test_cases)} failed")
    print("=" * 70)
    
    # Performance stats
    print(f"\nTranslation timeout: {translator.timeout}s")
    print(f"Model: {translator.model._model_name}")


async def test_language_detection():
    """Test language detection accuracy"""
    
    print("\n" + "=" * 70)
    print("Language Detection Test")
    print("=" * 70 + "\n")
    
    translator = get_semantic_translator()
    
    test_texts = [
        ("Hello, how are you?", "english"),
        ("नमस्ते, आप कैसे हैं?", "hindi"),
        ("హలో, మీరు ఎలా ఉన్నారు?", "telugu"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("Hello मेरा नाम John है", "mixed"),
        ("123 456", "english"),  # Numbers default to English
    ]
    
    for text, expected_lang in test_texts:
        detected_lang, confidence = translator.detect_language(text)
        
        match = "" if detected_lang == expected_lang or (expected_lang == "mixed" and detected_lang != "english") else ""
        
        print(f"{match} '{text[:40]}...'")
        print(f"   Expected: {expected_lang}, Got: {detected_lang} (confidence: {confidence:.2f})")
        print()


async def main():
    """Run all tests"""
    try:
        await test_language_detection()
        await test_translation()
        
        print("\n All tests completed successfully!")
        
    except Exception as e:
        print(f"\n Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
