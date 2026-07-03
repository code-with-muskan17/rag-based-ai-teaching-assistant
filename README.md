# 🎓 RAG-Based AI Teaching Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://code-with-muskan17-rag-based-ai-teaching-assistant-app-vlwkf.streamlit.app)

An AI-powered assistant that answers questions about course video lectures — telling you exactly which video covers a topic and at what timestamp.

Built end-to-end using Python, OpenAI Whisper, Google Gemini, and Streamlit.

---

## What It Does

Instead of scrubbing through hours of video to find a topic, just ask:

- *"Where is CSS Flexbox taught?"*
- *"Which video covers HTML forms?"*
- *"Where can I learn JavaScript arrays?"*

The assistant finds the most relevant video chunks and tells you exactly where to go.

---

## How It Works

**Offline Pipeline (run once to prepare data):**

```
Course Videos → MP3 Files → JSON Transcripts → Merged Chunks → embeddings.joblib
```

**Live Pipeline (every query):**

```
User Question → Gemini Embedding → Cosine Similarity Search → Top 5 Chunks → Gemini Answer
```

---

## Project Structure

```
rag-based-ai-teaching-assistant/
├── app.py                  # Streamlit app — entry point for deployment
├── preprocess_json.py      # Converts transcript chunks to Gemini embeddings
├── merge_chunks.py         # Groups Whisper segments into larger chunks
├── mp3_to_json.py          # Transcribes audio to JSON using Whisper
├── video_to_mp3.py         # Extracts MP3 from video files using ffmpeg
├── embeddings.joblib       # Pre-computed vector embeddings (ready to use)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Transcription | OpenAI Whisper — Hindi audio → English text |
| Embeddings | Google Gemini `embedding-001` |
| Vector Search | Scikit-learn cosine similarity |
| Answer Generation | Google Gemini `gemini-1.5-flash` |
| UI & Deployment | Streamlit |
| Data | Pandas + Joblib |

---

## Run It On Your Own Course Videos

### Step 1 — Add your videos
Place all video files inside the `videos/` folder.

### Step 2 — Extract audio
```bash
python video_to_mp3.py
```

### Step 3 — Transcribe
```bash
python mp3_to_json.py
```

### Step 4 — Merge chunks
```bash
python merge_chunks.py
```

### Step 5 — Generate embeddings
Add your Gemini API key to `preprocess_json.py`, then:
```bash
python preprocess_json.py
```

### Step 6 — Run the app
```bash
streamlit run app.py
```

---

## Deployment

The app is deployed on Streamlit Cloud. The Gemini API key is stored securely in Streamlit Secrets — not in the codebase.

To run locally, create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```

---

## Key Decisions

**Why Whisper?** Handles Hindi audio and translates to English automatically — no manual transcription needed.

**Why merge chunks?** Whisper segments are 1-2 seconds each. Merging every 5 segments gives better context for semantic search.

**Why Gemini for embeddings?** Free, reliable, and consistent — same model used for both indexing and querying ensures accurate similarity scores.

**Why cosine similarity?** Fast and effective for semantic search on dense vector embeddings without needing a vector database.

---

## Limitations

- Works on the course content it was trained on — unrelated questions are redirected
- Transcription quality depends on audio clarity
- Timestamp accuracy depends on Whisper segmentation

---

## Author

Muskan Agrawal — [code-with-muskan17](https://github.com/code-with-muskan17)
