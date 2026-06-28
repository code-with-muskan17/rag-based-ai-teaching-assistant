import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.joblib")

df = load_embeddings()


def get_query_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def generate_answer(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def search_and_answer(question):
    query_embedding = get_query_embedding(question)

    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [query_embedding]
    ).flatten()

    top_indices = similarities.argsort()[::-1][:5]
    relevant_chunks = df.loc[top_indices]

    prompt = f"""You are an AI assistant helping students navigate a web development course.
Here are the most relevant video subtitle chunks for the user's question.
Each chunk contains: video number, title, start time (seconds), end time (seconds), and transcript text.

{relevant_chunks[["number", "title", "start", "end", "text"]].to_json(orient="records")}

---------------------------------------
User question: "{question}"

Answer in a friendly, conversational way. Tell the user which video covers this topic
and at what timestamp they can find it. Guide them to go to that specific video.
Do not mention the JSON format or chunk structure — just answer naturally.
If the question has nothing to do with web development, politely let them know
you can only help with questions related to this course.
"""

    answer = generate_answer(prompt)
    return answer, relevant_chunks


st.markdown("# 🎓 AI Teaching Assistant")
st.markdown("Ask anything about the course — I'll point you to the right video and timestamp.")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 💬 Your Question")

    examples = [
        "Where is media taught in this course?",
        "Which video covers HTML tables?",
        "Where can I learn CSS selectors?",
        "Which video explains JavaScript arrays?",
        "Where is flexbox taught?",
    ]

    selected_example = st.selectbox(
        "Or pick an example question:",
        [""] + examples
    )

    question = st.text_area(
        "Type your question:",
        value=selected_example,
        placeholder="e.g. Where is CSS Flexbox taught?",
        height=120
    )

    submitted = st.button("🔍 Find Answer", type="primary", use_container_width=True)


with col_right:
    st.markdown("### 📖 Answer")

    if submitted and question.strip():
        with st.spinner("Searching through course videos..."):
            try:
                answer, relevant_chunks = search_and_answer(question)

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border-left: 4px solid #4CAF50;
                        border-radius: 8px;
                        padding: 20px;
                        margin-bottom: 20px;
                    ">
                        <p style="color: #e0e0e0; font-size: 16px; line-height: 1.8; margin: 0;">
                            {answer}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### 📌 Relevant Video Chunks")
                display_df = relevant_chunks[["number", "title", "start", "end"]].copy()
                display_df.columns = ["Video #", "Title", "Start (sec)", "End (sec)"]
                display_df = display_df.reset_index(drop=True)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    elif submitted and not question.strip():
        st.warning("Please type a question first.")

    else:
        st.markdown(
            """
            <div style="
                background: #1e1e2e;
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 40px;
                text-align: center;
            ">
                <p style="color: #888; font-size: 18px; margin: 0;">
                    👈 Type your question and click <strong>Find Answer</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


with st.sidebar:
    st.markdown("## 🎓 About")
    st.markdown("""
    This AI Teaching Assistant helps you navigate
    a web development course using RAG.

    Ask any topic and it will tell you:
    - 📹 Which video covers it
    - ⏱️ Exact timestamp to jump to
    - 📝 What's taught at that point
    """)

    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. Your question is converted to a vector embedding
    2. Compared against all video transcript chunks
    3. Top 5 most relevant chunks are retrieved
    4. Gemini generates a human-friendly answer
    """)

    st.markdown("---")
    st.markdown("### 📊 Course Info")
    st.metric("Total Video Chunks", len(df))
    st.caption("Powered by Gemini AI + RAG Pipeline")
