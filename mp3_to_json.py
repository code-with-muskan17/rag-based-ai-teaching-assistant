import whisper
import json
import os

model = whisper.load_model("base")

audios = os.listdir("audios")

for audio in audios:
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(f"\nProcessing: {audio} | Video #{number} | Title: {title}")

        print("Transcribing...")
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )
        print(f"Transcription done for: {audio}")

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, indent=4)

        print(f"Saved JSON: jsons/{audio}.json")