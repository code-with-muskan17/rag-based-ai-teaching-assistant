import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    h1 a, h2 a, h3 a, h4 a { display: none !important; }

    div.stButton > button[kind="primary"] {
        background: #2d6a4f;
        border: none;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 8px;
        padding: 14px;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #1a472a;
        border: none;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        color: #888 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.joblib")

df = load_embeddings()


def get_query_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


def generate_answer(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
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


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <p style="color: #4CAF50; font-size: 11px; letter-spacing: 1.5px;
              margin: 0 0 16px 0; font-weight: 600;">
        POWERED BY GEMINI AI + RAG
    </p>

    <p style="color: #aaa; font-size: 13px; line-height: 1.8; margin: 0 0 20px 0;">
        Instead of scrubbing through hours of video to find a topic —
        just ask a question and get pointed to the exact video and timestamp.
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="background: #1a1a2e; border-radius: 10px; padding: 16px;
                border: 1px solid #2a2a3d; margin-bottom: 20px;">
        <p style="color: #666; font-size: 11px; letter-spacing: 1.5px;
                  font-weight: 600; margin: 0 0 14px 0;">HOW IT WORKS</p>
        <div style="display:flex; align-items:flex-start; margin-bottom:12px;">
            <span style="color:#4CAF50; font-size:14px; margin-right:10px; line-height:1.4;">①</span>
            <p style="color:#bbb; font-size:13px; margin:0; line-height:1.5;">
                Question converted to a vector embedding</p>
        </div>
        <div style="display:flex; align-items:flex-start; margin-bottom:12px;">
            <span style="color:#4CAF50; font-size:14px; margin-right:10px; line-height:1.4;">②</span>
            <p style="color:#bbb; font-size:13px; margin:0; line-height:1.5;">
                Compared against all transcript chunks</p>
        </div>
        <div style="display:flex; align-items:flex-start; margin-bottom:12px;">
            <span style="color:#4CAF50; font-size:14px; margin-right:10px; line-height:1.4;">③</span>
            <p style="color:#bbb; font-size:13px; margin:0; line-height:1.5;">
                Top 5 relevant chunks retrieved</p>
        </div>
        <div style="display:flex; align-items:flex-start;">
            <span style="color:#4CAF50; font-size:14px; margin-right:10px; line-height:1.4;">④</span>
            <p style="color:#bbb; font-size:13px; margin:0; line-height:1.5;">
                Gemini generates a human-friendly answer</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color: #666; font-size: 11px; letter-spacing: 1.5px;
              font-weight: 600; margin: 0 0 12px 0;">COURSE INFO</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Video Chunks", len(df))
    with col2:
        st.metric("Videos", 3)


# ── Main Area ──
st.markdown("# 🎓 AI Teaching Assistant")
st.markdown("Ask anything about the course — I'll point you to the right video and timestamp.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 💬 Your Question")

    examples = [
        "Where is media taught in this course?",
        "Which video covers HTML video tags?",
        "Where is SEO taught?",
        "What is covered in the SEO video?",
        "Where are audio tags explained?",
        "What does video 10 cover?",
        "Where is Core Web Vitals explained?",
    ]

    selected_example = st.selectbox(
        "Pick an example or type your own:",
        ["— select an example —"] + examples
    )

    if selected_example == "— select an example —":
        selected_example = ""

    question = st.text_area(
        "Your question:",
        value=selected_example,
        placeholder="e.g. Where is SEO taught in this course?",
        height=130,
        label_visibility="collapsed"
    )

    submitted = st.button("Find Answer", type="primary", use_container_width=True)


with col_right:
    st.markdown("### 📖 Answer")

    if submitted and question.strip():
        with st.spinner("Searching through course videos..."):
            try:
                answer, relevant_chunks = search_and_answer(question)

                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid #2d6a4f;
                        background: #1a1a2e;
                        border-radius: 0 8px 8px 0;
                        padding: 20px 24px;
                        margin-bottom: 24px;
                    ">
                        <p style="color: #e0e0e0; font-size: 15px; line-height: 1.9; margin: 0;">
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
                border: 2px dashed #2a2a3d;
                border-radius: 10px;
                padding: 50px 32px;
                text-align: center;
                margin-top: 4px;
            ">
                <p style="color: #444; font-size: 32px; margin: 0 0 12px 0;">🔍</p>
                <p style="color: #666; font-size: 15px; margin: 0; line-height: 1.7;">
                    Pick an example or type your question<br>
                    then click <strong style="color: #4CAF50;">Find Answer</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
