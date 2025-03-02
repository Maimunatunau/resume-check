import streamlit as st
import pdfplumber
import spacy
import language_tool_python
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Function to extract text from Word documents
def extract_text_from_word(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract key phrases (skills, qualifications, tools) from text
def extract_key_phrases(text):
    doc = nlp(text.lower())
    key_phrases = {chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1}
    return key_phrases

# Function to filter relevant phrases
def filter_relevant_phrases(phrases):
    ignore_words = {"someone", "we", "the ability", "tasks", "bonus points", "experience", "various locations",
                    "familiarity", "ability", "a proactive and solution-oriented approach", "governance",
                    "escalation management", "contact/call centre experience", "strong organisational and project coordination skills"}
    filtered_phrases = {phrase for phrase in phrases if phrase not in ignore_words and not phrase.startswith("a ")}
    return filtered_phrases

# Function to calculate resume-job description match score
def compare_resume_to_job(resume_text, job_desc_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_desc_text])
    return cosine_similarity(vectorizer.toarray())[0, 1] * 100

# Function to analyze readability
def calculate_readability(text):
    words = len(re.findall(r'\w+', text))
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    score = max(0, 100 - (words / sentences * 2))
    return score

# Function to check ATS-friendly formatting
def check_resume_format(text):
    if any(tag in text.lower() for tag in ["image", "table", "text box"]):
        return "⚠️ Your resume may contain non-ATS-friendly elements like images or text boxes."
    return "✅ Your resume is ATS-friendly."

# Streamlit UI
st.title('🚀 AI-Powered Resume Analyzer')

uploaded_file = st.file_uploader("📄 Upload Resume (PDF or Word)", type=["pdf", "docx"])
job_desc = st.text_area("✍️ Paste Job Description", height=100)

if st.button("🔍 Analyze Resume"):
    if uploaded_file and job_desc:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_word(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a PDF or Word document.")
            resume_text = ""
        
        if resume_text:
            resume_phrases = extract_key_phrases(resume_text)
            job_phrases = extract_key_phrases(job_desc)
            missing_phrases = filter_relevant_phrases(job_phrases - resume_phrases)
            match_score = compare_resume_to_job(resume_text, job_desc)
            readability_score = calculate_readability(resume_text)
            ats_feedback = check_resume_format(resume_text)
            
            # Display results
            st.subheader("📊 Resume Analysis")
            st.write(f"🔹 **Match Score:** {match_score:.2f}%")
            if match_score > 70:
                st.success("✅ Your resume is a strong match for this job!")
            else:
                st.warning("⚠️ Consider aligning your experience more closely with the job requirements.")
            
            if missing_phrases:
                st.subheader("🔍 Suggested Keywords & Skills to Add")
                st.write("- " + "\n- ".join(missing_phrases))
            
            st.subheader("📖 Readability Score")
            st.write(f"Your resume readability score: **{readability_score:.2f}**")
            if readability_score < 60:
                st.warning("⚠️ Your resume may be too complex. Consider simplifying your sentences.")
            elif readability_score > 80:
                st.success("✅ Your resume is easy to scan!")
            else:
                st.info("⚠️ Your resume has moderate complexity. Consider slight refinements.")
            
            st.subheader("📑 ATS Compatibility")
            st.write(ats_feedback)
