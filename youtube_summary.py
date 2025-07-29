import os
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import requests
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import textwrap

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Missing GOOGLE_API_KEY in .env")
    st.stop()

# Configure page
st.set_page_config(page_title="📽️ YouTube Transcript Summarizer", layout="centered")

# Configure Gemini model
genai.configure(api_key=api_key)
gen_config = GenerationConfig(temperature=0.2, max_output_tokens=4096)
models = genai.list_models()
model_names = [m.name for m in models if "generateContent" in m.supported_generation_methods]
default = "models/gemini-1.5-flash-latest"
default_index = model_names.index(default) if default in model_names else 0
selected_model = st.sidebar.selectbox("🧠 Choose model:", model_names, index=default_index)

model = genai.GenerativeModel(
    model_name=selected_model,
    generation_config=gen_config
)

# --- Helper functions ---

def get_video_id(url: str) -> str:
    p = urlparse(url)
    query = parse_qs(p.query)
    return query.get("v", [None])[0]

def clean_youtube_url(url: str) -> str:
    p = urlparse(url)
    video_id = parse_qs(p.query).get("v", [None])[0]
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"

def get_transcript_from_url(url: str) -> str:
    video_id = get_video_id(url)
    if not video_id:
        raise RuntimeError("❌ Invalid YouTube URL — no video ID found.")

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcripts.find_transcript(['en'])
        except NoTranscriptFound:
            transcript = list(transcripts)[0]

        segments = transcript.fetch()
        if not segments:
            raise RuntimeError("❌ Transcript is empty or unreadable.")

        st.success(f"✅ Transcript language: `{transcript.language_code}`")
        return " ".join(seg.text for seg in segments)

    except TranscriptsDisabled:
        raise RuntimeError("❌ Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("❌ No transcript found in any supported language.")
    except requests.exceptions.RequestException:
        raise RuntimeError("❌ Network error or YouTube blocked the request.")
    except Exception as e:
        msg = str(e).lower()
        if "could not retrieve a transcript" in msg:
            raise RuntimeError("❌ YouTube does not provide transcripts for this video.")
        elif "no transcripts are available" in msg:
            raise RuntimeError("❌ No available transcript for this video in any language.")
        elif "list index out of range" in msg:
            raise RuntimeError("❌ No transcripts found. Video may be private, unlisted, or region-locked.")
        elif "parse error" in msg or "malformed" in msg:
            raise RuntimeError("❌ YouTube returned unreadable transcript data.")
        elif "no element found" in msg:
            raise RuntimeError("❌ YouTube returned an empty response. Video likely has no transcript.")
        else:
            raise RuntimeError(f"❌ Failed to fetch transcript: {str(e)}")

def split_text_into_chunks(text, max_tokens=3000):
    """Splits long text into smaller chunks."""
    return textwrap.wrap(text, width=max_tokens, break_long_words=False)

def generate_summary(text: str) -> str:
    """Generate enriched summary using Gemini model with prompt."""
    prompt = (
        "You are a knowledgeable and precise educational AI. A technical YouTube transcript is provided.\n\n"
        "Your job is to generate a well-structured, human-friendly enriched summary in **Markdown** with clean **LaTeX** math formatting.\n\n"
        "**Output Structure:**\n"
        "### 1. Overview\n"
        "- List all key topics discussed in bullet points.\n\n"
        "### 2. Detailed Explanation\n"
        "- For each topic:\n"
        "  - Use proper headings and subheadings.\n"
        "  - Define key terms and formulas.\n"
        "  - Show clean math using `$$...$$`.\n"
        "  - Use clean matrix formatting:\n"
        "    $$\n"
        "    \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}\n"
        "    $$\n"
        "  - Provide full examples.\n\n"
        "### 3. Extra Notes\n"
        "- Add assumptions, tips, or real-world connections.\n\n"
        f"Transcript:\n{text}"
    )
    response = model.generate_content(prompt)
    return response.text

def generate_chunked_summary(full_transcript: str) -> str:
    """Handle long transcript summarization by chunking."""
    chunks = split_text_into_chunks(full_transcript, max_tokens=3000)
    all_summaries = []
    for idx, chunk in enumerate(chunks):
        with st.spinner(f"📦 Summarizing chunk {idx+1}/{len(chunks)}..."):
            summary = generate_summary(chunk)
            all_summaries.append(summary)
    # Optionally merge all summaries into one final summary
    final_summary_prompt = (
        "You are given multiple partial summaries generated from chunks of a full transcript. "
        "Merge them into a single, cohesive enriched summary in Markdown with proper headings, examples, and LaTeX formatting."
        f"\n\nSummaries:\n{''.join(all_summaries)}"
    )
    with st.spinner("🔗 Merging all partial summaries..."):
        final_summary = model.generate_content(final_summary_prompt)
    return final_summary.text

def get_video_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

# --- Streamlit UI ---
st.title("📽️ YouTube Transcript to Enriched Summary")

video_url = st.text_input("🔗 Enter YouTube Video Link:")

if video_url:
    if "list=" in video_url:
        st.warning("ℹ️ You're using a playlist URL. For best results, use a direct video link.")

    try:
        clean_url = clean_youtube_url(video_url)
        if not clean_url:
            st.error("❌ Could not extract video ID from the URL.")
            st.stop()

        with st.spinner("⏳ Fetching transcript..."):
            vid = get_video_id(clean_url)
            st.write(f"🔎 Video ID: `{vid}`")
            transcript = get_transcript_from_url(clean_url)
            thumbnail_url = get_video_thumbnail(vid)
            st.image(thumbnail_url, use_container_width=True)

        enriched_summary = generate_chunked_summary(transcript)

        st.subheader("🧠 Enriched Summary with Examples")
        st.markdown(enriched_summary, unsafe_allow_html=True)

    except Exception as err:
        st.error(str(err))


















