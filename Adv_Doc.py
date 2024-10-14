import streamlit as st
import re
import nltk
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import docx
import pdfplumber
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
from fuzzywuzzy import fuzz
import difflib
import time
from collections import defaultdict
from transformers import logging

# Suppress transformers warnings
logging.set_verbosity_error()

# Load models
clause_model_name = "paraphrase-MiniLM-L6-v2"
logical_model_name = "roberta-large-mnli"
topic_model_name = "all-mpnet-base-v2"

clause_model = SentenceTransformer(clause_model_name)
logical_tokenizer = AutoTokenizer.from_pretrained(logical_model_name)
logical_model = AutoModelForSequenceClassification.from_pretrained(logical_model_name, ignore_mismatched_sizes=True)
topic_model = SentenceTransformer(topic_model_name)

# Simple password-based authentication
def check_password():
    password = st.text_input("Enter the password", type="password")
    if password == "Cloud@Compare11#a":
        return True
    else:
        if password:
            st.warning("Please enter a valid password")
        return False

# Password gate for the app
if check_password():
    # Load the provided logo
    logo_path = "Logo_For white or light backgrounds.png"
    logo_image = Image.open(logo_path)

    # Set custom styling using colors from the logo
    primary_color = "#1B75BC"
    secondary_color = "#8AC640"
    st.markdown(f"""
        <style>
        .reportview-container {{
            background: linear-gradient(90deg, {secondary_color} 30%, white 30%, white 70%, {secondary_color} 70%);
        }}
        .sidebar .sidebar-content {{
            background: {primary_color};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {primary_color};
        }}
        </style>
        """, unsafe_allow_html=True)

    # Display the logo and title
    st.image(logo_image, width=300)
    st.title("Clarence & Partners Document Preprocessing and Comparison App")

    # Add introductory text and disclaimer
    st.markdown("""
    ### This app is used to preprocess and compare documents in advance of seeking outputs from the ChatGPT Custom GPT from Clarence & Partners.

    **Disclaimer**: No output provided by this tool should be treated as legal advice and users are encouraged to seek advice of specialist legal counsel.
    """)

    # Use caching to download NLTK data once
    @st.cache_resource
    def download_nltk_data():
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Call the function to download NLTK data
    download_nltk_data()

    # Function to read DOCX files
    def read_docx(file):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    # Function to read PDF files
    def read_pdf(file):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    # Function to extract text from a document
    def extract_text(file):
        if file.type == "application/pdf":
            return read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx(file)
        else:
            return file.read().decode("utf-8", errors='replace')

    # Function to normalize and preprocess text
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    # Function to preprocess and tokenize text into sentences and words
    @st.cache_data
    def preprocess_sentences(text):
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        processed = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word.lower() not in stop_words]
            processed_sentence = ' '.join(tokens)
            processed.append(processed_sentence)
        return sentences, processed

    # Function to filter sentences based on focus terms with fuzzy matching
    def filter_sentences(sentences, focus_terms, threshold):
        filtered_sentences = []
        for sentence in sentences:
            for term in focus_terms:
                if fuzz.partial_ratio(term.lower(), sentence.lower()) >= threshold:
                    filtered_sentences.append(sentence)
                    break
        return filtered_sentences

    # Function to calculate semantic similarity using SentenceTransformer
    def calculate_semantic_similarity(clause, sentences, model):
        clause_embeddings = model.encode([clause], convert_to_tensor=True)
        sentences_embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(clause_embeddings, sentences_embeddings)[0]
        return {sentences[i]: float(cosine_scores[i]) for i in range(len(sentences))}

    # Function to align sentences between documents using combined cosine similarity and fuzzy matching
    def align_sentences_combined(sentences1, sentences2, threshold=0.75, cosine_weight=0.5, fuzzy_weight=0.5):
        vectorizer = lambda s: np.array([1 if w in s else 0 for w in set(" ".join(sentences1 + sentences2).split())])
        sentence_vectors1 = np.array([vectorizer(sentence) for sentence in sentences1])
        sentence_vectors2 = np.array([vectorizer(sentence) for sentence in sentences2])

        similarity_matrix = cosine_similarity(sentence_vectors1, sentence_vectors2)

        aligned_pairs = []
        for idx1, sentence1 in enumerate(sentences1):
            best_combined_score = 0
            best_match_sentence = None
            best_cosine_score = 0
            best_fuzzy_score = 0

            for idx2, sentence2 in enumerate(sentences2):
                cosine_score = similarity_matrix[idx1][idx2]
                fuzzy_score = fuzz.ratio(sentence1, sentence2) / 100.0
                combined_score = (cosine_weight * cosine_score) + (fuzzy_weight * fuzzy_score)

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_match_sentence = sentence2
                    best_cosine_score = cosine_score
                    best_fuzzy_score = fuzzy_score

            if best_combined_score >= threshold:
                aligned_pairs.append({
                    'doc1_sentence': sentence1,
                    'doc2_sentence': best_match_sentence,
                    'combined_similarity_score': best_combined_score,
                    'cosine_similarity_score': best_cosine_score,
                    'fuzzy_similarity_score': best_fuzzy_score
                })
            else:
                aligned_pairs.append({
                    'doc1_sentence': sentence1,
                    'doc2_sentence': None,
                    'combined_similarity_score': best_combined_score,
                    'cosine_similarity_score': best_cosine_score,
                    'fuzzy_similarity_score': best_fuzzy_score
                })
        return aligned_pairs

    # Main Streamlit app function
    st.title("Advanced Document Comparison and Analysis Tool")

    # Upload CSV file containing focus terms/clauses
    focus_terms_file = st.file_uploader("Upload CSV file containing focus terms/clauses (with alternate terms)", type=['csv'])

    # Upload sets of documents for comparison
    doc1_files = st.file_uploader("Upload first document set", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    doc2_files = st.file_uploader("Upload second document set", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    # Add sliders to adjust thresholds and weights
    fuzzy_threshold = st.slider("Set Fuzzy Matching Threshold", min_value=0, max_value=100, value=60)
    similarity_threshold = st.slider("Set Similarity Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    cosine_weight = st.slider('Cosine Similarity Weight', min_value=0.0, max_value=1.0, value=0.5)
    fuzzy_weight = st.slider('Fuzzy Matching Weight', min_value=0.0, max_value=1.0, value=0.5)

    # Button to trigger the app
    generate_report = st.button("Generate Report")

    if generate_report and doc1_files and doc2_files and focus_terms_file:
        # Read the focus terms from CSV
        focus_terms_df = pd.read_csv(focus_terms_file)
        st.write("Loaded CSV Terms:")
        st.dataframe(focus_terms_df)

        # Extract text from document sets
        doc1_texts = [extract_text(file) for file in doc1_files]
        doc2_texts = [extract_text(file) for file in doc2_files]

        # Combine texts into one string per set
        doc1_combined = "\n".join(doc1_texts)
        doc2_combined = "\n".join(doc2_texts)

        # Split documents into individual sentences for efficient filtering
        doc1_sentences = re.split(r'(?<=[.!?]) +', doc1_combined)
        doc2_sentences = re.split(r'(?<=[.!?]) +', doc2_combined)

        # Extract focus terms from CSV
        primary_terms = focus_terms_df['Primary Term'].tolist()
        alternate_terms = focus_terms_df['Alternate Terms'].tolist()
        all_terms = primary_terms + [alt for alt in alternate_terms if isinstance(alt, str)]

        # Filter sentences that contain focus terms using fuzzy matching
        doc1_sentences_filtered = filter_sentences(doc1_sentences, all_terms, fuzzy_threshold)
        doc2_sentences_filtered = filter_sentences(doc2_sentences, all_terms, fuzzy_threshold)

        # Compare filtered sentences using semantic similarity
        st.write("Calculating semantic similarity for filtered sentences...")
        clause_analysis_results = []

        for clause_1 in doc1_sentences_filtered:
            for clause_2 in doc2_sentences_filtered:
                similarity_scores = calculate_semantic_similarity(clause_1, [clause_2], clause_model)
                score = similarity_scores[clause_2]
                if score > similarity_threshold:
                    clause_analysis_results.append({
                        "Document Set 1 Clause": clause_1,
                        "Document Set 2 Clause": clause_2,
                        "Similarity Score": score
                    })

        # Create DataFrame from the clause analysis results
        if clause_analysis_results:
            structured_report = pd.DataFrame(clause_analysis_results)

            # Display results and provide download button
            st.write("### Clause Analysis Results")
            st.dataframe(structured_report)

            st.download_button(
                label="Download Analysis Report",
                data=structured_report.to_csv(index=False),
                file_name="filtered_clause_comparison_report.csv",
                mime="text/csv"
            )
        else:
            st.warning("No matching clauses found. Try adjusting the similarity threshold or fuzzy matching settings.")
    else:
        st.warning("Please upload the required files (documents and CSV).")
else:
    st.stop()