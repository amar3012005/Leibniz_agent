"""
Quick microphone test to verify audio capture is working
"""
import sounddevice as sd
import numpy as np
import time

print("=" * 70)
print("Microphone Audio Level Test")
print("=" * 70)
print("\n🎤 Available Input Devices:")
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        marker = ">" if i == sd.default.device[0] else " "
        print(f"{marker} {i}: {dev['name']} ({dev['max_input_channels']} channels)")

print(f"\n📍 Using default device: {sd.default.device[0]}")
print("\n🔊 Speak into your microphone now...")
print("   Watching audio levels for 5 seconds...\n")

def audio_callback(indata, frames, time_info, status):
    """Monitor audio levels"""
    if status:
        print(f"Status: {status}")
    
    # Calculate RMS (root mean square) amplitude
    rms = np.sqrt(np.mean(indata**2))
    # Calculate peak amplitude
    peak = np.max(np.abs(indata))
    
    # Visual level meter
    bar_length = int(rms * 100)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    
    print(f"\rLevel: {bar} RMS: {rms:.4f} Peak: {peak:.4f}", end="", flush=True)

try:
    # Record for 5 seconds
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        time.sleep(5)
    
    print("\n\n✅ Microphone test complete!")
    print("\nExpected behavior:")
    print("  • Silence: RMS ~0.0001-0.01, Peak ~0.01-0.05")
    print("  • Speech: RMS ~0.05-0.3, Peak ~0.3-0.9")
    print("\nIf you saw NO movement in the level bar:")
    print("  1. Check microphone privacy settings in Windows")
    print("  2. Ensure microphone is not muted")
    print("  3. Try a different input device (set AUDIO_INPUT_DEVICE env var)")

except Exception as e:
    print(f"\n\n❌ Error: {e}")
    print("\nThis might indicate:")
    print("  • Microphone is in use by another application")
    print("  • Driver issues")
    print("  • Permission problems")
