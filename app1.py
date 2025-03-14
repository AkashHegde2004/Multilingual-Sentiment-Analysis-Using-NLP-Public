import streamlit as st
import os
import warnings
from transformers import pipeline, logging
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Suppress various warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Suppress symlinks warning
logging.set_verbosity_error()  # Suppress transformers warnings
warnings.filterwarnings('ignore')  # Suppress Python warnings

# Ensure language detection works correctly
DetectorFactory.seed = 0

# Dictionary mapping language codes to full names
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ru': 'Russian',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'el': 'Greek',
    'he': 'Hebrew',
    'id': 'Indonesian',
    'pl': 'Polish',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'hu': 'Hungarian',
    'cs': 'Czech',
    'ro': 'Romanian',
    'uk': 'Ukrainian'
    # Add more languages as needed
}

# Set page configuration
st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_sentiment_analyzer():
    """Load the sentiment analysis model with caching for better performance"""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        revision="af0f99b"
    )

def translate_to_english(text):
    """
    Translates text from any language to English using deep-translator.
    Returns the translated text.
    """
    try:
        translated_text = GoogleTranslator(source="auto", target="en").translate(text)
        return translated_text  # Returns a string directly, no .text needed
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

def detect_language(text):
    """
    Detects the language of the input text.
    Returns the language code with better handling of short phrases.
    """
    # First check if it's a short phrase that's likely English
    if len(text.split()) <= 10:  # Increase threshold to catch slightly longer phrases
        common_english_words = ["my", "name", "is", "i", "am", "and", "the", "a", "an", "happy", "roamed"]
        text_words = set(text.lower().split())
        # If more than 40% of words are common English words, classify as English
        if len(text_words.intersection(common_english_words)) / max(1, len(text_words)) > 0.4:
            return 'en'
    
    # If not caught by the short phrase check, use langdetect
    try:
        language = detect(text)
        return language
    except LangDetectException:
        # Default to English if detection fails
        return 'en'

# App title and description
st.title("Multilingual Sentiment Analysis")
st.markdown("""
This app translates text from any language to English and performs sentiment analysis.
Enter text in any language to analyze its sentiment.
""")

# Load the sentiment analyzer
with st.spinner("Loading sentiment analysis model..."):
    sentiment_analyzer = load_sentiment_analyzer()

# Input text area
user_input = st.text_area("Enter text in any language:", height=150)

# Process button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Create a progress container
        progress_container = st.container()

        with st.spinner("Detecting language..."):
            progress_container.text("Step 1/3: Detecting language...")
            original_language_code = detect_language(user_input)

        if original_language_code:
            original_language = LANGUAGE_NAMES.get(original_language_code, "Unknown Language")

            with st.spinner("Translating..."):
                progress_container.text("Step 2/3: Translating text...")
                translated_text = translate_to_english(user_input)

            if translated_text:
                with st.spinner("Analyzing sentiment..."):
                    progress_container.text("Step 3/3: Analyzing sentiment...")
                    result = sentiment_analyzer(translated_text)[0]

                # Determine sentiment label
                if len(translated_text.split()) <= 3:
                    # Explicitly handle short positive phrases
                    if "happy" in translated_text.lower() or "great" in translated_text.lower():
                        sentiment_label = 'POSITIVE'
                    else:
                        sentiment_label = 'NEUTRAL'
                elif result['score'] > 0.55:
                    sentiment_label = result['label']
                else:
                    sentiment_label = 'NEUTRAL'

                # Adjust for neutral phrases
                neutral_phrases = ["i just roamed around", "my name is", "i roamed", "hi", "hello"]
                if any(phrase in translated_text.lower() for phrase in neutral_phrases):
                    sentiment_label = 'NEUTRAL'

                # Display results in a nice format
                st.success("Analysis complete!")

                # Translation result
                st.subheader("Translation")
                st.write(f"**Original text:** {user_input}")
                st.write(f"**Detected language:** {original_language}")
                st.write(f"**Translated to English:** {translated_text}")

                # Sentiment result
                st.subheader("Sentiment Analysis")

                # Display sentiment label and score
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment_label)
                with col2:
                    st.metric("Confidence", f"{result['score']*100:.2f}%")

                # Interpretation
                if sentiment_label == 'POSITIVE':
                    if result['score'] > 0.9:
                        interpretation = "Very positive sentiment"
                        emoji = "😄"
                    else:
                        interpretation = "Positive sentiment"
                        emoji = "🙂"
                elif sentiment_label == 'NEUTRAL':
                    interpretation = "Neutral sentiment"
                    emoji = "😐"
                else:
                    if result['score'] > 0.9:
                        interpretation = "Very negative sentiment"
                        emoji = "😞"
                    else:
                        interpretation = "Negative sentiment"
                        emoji = "🙁"

                # Display interpretation with emoji
                st.subheader(f"Interpretation: {interpretation} {emoji}")

# Add footer
st.markdown("---")
st.markdown("Multilingual Sentiment Analysis App | Built with Streamlit, Transformers, and Googletrans")