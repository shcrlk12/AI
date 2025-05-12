import whisper

model = whisper.load_model("turbo")  # tiny, base, small, medium, large
result = model.transcribe("sample2.m4a", language="en")

for segment in result["segments"]:
    print(f"[{segment['start']:.2f} ~ {segment['end']:.2f}] {segment['text']}")