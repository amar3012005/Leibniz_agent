"""
LemonFox AI TTS Test - Simple HTTP Request
API Docs: https://www.lemonfox.ai/apis/text-to-speech
"""

import requests
import json

# Hardcoded API key (from your script)
API_KEY = "gm5Bn9DssB4Jpe6MFJCic6Lv1IaAYA11"
API_URL = "https://api.lemonfox.ai/v1/audio/speech"

print(" Testing LemonFox AI Text-to-Speech API")
print("=" * 70)

# Request parameters
payload = {
    "input": "Hallo! Das ist ein Test der LemonFox KI Text-zu-Sprache. Es klingt ziemlich natürlich!",
    "voice": "sarah",  # Options: heart, bella, michael, sarah, etc.
    "response_format": "mp3",  # mp3, wav, opus, flac
    "speed": 1.0,  # 0.25 to 4.0
    "language": "de-de"  # German! (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print(f" Text: {payload['input']}")
print(f" Voice: {payload['voice']}")
print(f" Format: {payload['response_format']}")
print(f"\n⏳ Sending request to LemonFox API...")

try:
    # Make API request
    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    # Check response
    if response.status_code == 200:
        # Save audio file
        output_file = "lemonfox_speech.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f" SUCCESS!")
        print(f" Audio saved to: {output_file}")
        print(f" File size: {len(response.content)} bytes ({len(response.content)/1024:.1f} KB)")
        print(f"\n Play with: powershell -c 'Start-Process {output_file}'")
        
        # Auto-play (optional)
        import os
        print(f"\n Playing audio...")
        os.system(f'powershell -c "Start-Process {output_file}"')
        
    else:
        print(f" API Error!")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.Timeout:
    print(" Request timeout - API took too long to respond")
except requests.exceptions.RequestException as e:
    print(f" Request failed: {e}")
except Exception as e:
    print(f" Unexpected error: {e}")

print("\n" + "=" * 70)