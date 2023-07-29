from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

# Function to calculate sentiment score
def score(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity

# Function to classify sentiment
def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Set custom CSS styles
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #001f3f;
        color: white;
    }
    .header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #17a2b8;
    }
    .expander {
        background-color: #111d2b;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .expander-title {
        color: #17a2b8;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .text-input {
        background-color: #001f3f;
        color: white;
    }
    .text-input input {
        color: white;
    }
    .cleaned-text {
        white-space: pre-wrap;
        color: #17a2b8;
    }
    .results-table {
        background-color: #111d2b;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        padding: 1rem;
    }
    .download-button {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set header
st.markdown("<h1 class='header'>Sentiment Analysis</h1>", unsafe_allow_html=True)

# Analyze Text expander
with st.expander('Analyze Text', expanded=True):
    st.markdown("<h2 class='expander-title'>Analyze Text</h2>", unsafe_allow_html=True)
    # Text input for user input
    text = st.text_input('Text here:', value='', max_chars=None, key=None, type='default', help=None, on_change=None, args=None)
    if text:
        # Perform sentiment analysis on the text
        blob = TextBlob(text)
        st.write('Polarity:', round(blob.sentiment.polarity, 2))
        st.write('Subjectivity:', round(blob.sentiment.subjectivity, 2))

    # Pre-cleaned Text input
    pre = st.text_input('Clean Text:', value='', max_chars=None, key=None, type='default', help=None, on_change=None, args=None)
    if pre:
        # Clean the text using cleantext package
        cleaned_text = cleantext.clean(
            pre, clean_all=False, extra_spaces=True,
            stopwords=True, lowercase=True, numbers=True, punct=True
        )
        st.markdown(f"<div class='cleaned-text'>{cleaned_text}</div>", unsafe_allow_html=True)

# Analyze CSV expander
with st.expander('Analyze CSV', expanded=False):
    st.markdown("<h2 class='expander-title'>Analyze CSV</h2>", unsafe_allow_html=True)
    # File upload button for CSV
    upl = st.file_uploader('Upload file')

    if upl:
        # Read CSV file and perform sentiment analysis
        df = pd.read_csv(upl, encoding='latin1')  # Specify the appropriate encoding

        # Check if 'Unnamed: 0' column exists before deleting it
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.markdown("<div class='results-table'>", unsafe_allow_html=True)
        st.write(df.head(10))
        st.markdown("</div>", unsafe_allow_html=True)

        # Function to convert DataFrame to CSV
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        # Convert DataFrame to CSV and create download button
        csv = convert_df(df)
        st.markdown("<div class='download-button'>", unsafe_allow_html=True)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
        st.markdown("</div>", unsafe_allow_html=True)
