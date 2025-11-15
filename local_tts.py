from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", gpu=True)
tts.tts_to_file(text="Hello world!", file_path="output.wav")
