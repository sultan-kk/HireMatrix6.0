"""
Super TalentMatch AI — Unified HR Screener & Extractor
======================================================
This app integrates document extraction (Name, Email, Phone, Skills) 
and AI Candidate Screening (Match Score, Missing Skills, Questions) 
into a single, powerful Groq API call for maximum accuracy.
"""

import io
import json
import os
import re

import pandas as pd
import streamlit as st
from groq import Groq

# ===========================================================================
# CONFIGURATION
# ===========================================================================
APP_NAME = "Super TalentMatch AI"
APP_TAGLINE = "Unified Resume Extraction & Deep LLM Screening"
# Using a reliable Groq model for structured JSON output
GROQ_MODEL = "llama3-70b-8192" 
ACCEPTED_TYPES = ["pdf", "docx"]

# ===========================================================================
# PAGE CONFIG & CSS
# ===========================================================================
st.set_page_config(
    page_title=f"{APP_NAME} | HR Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    color: #1E293B; 
}
.stApp { background: #F8FAFC; }

.tm-header {
    padding: 1.6rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.4rem;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
}
.tm-header h1 { color: #FFFFFF; font-size: 2.1rem; font-weight: 800; margin: 0; }
.tm-header p { color: #94A3B8; margin-top: 0.3rem; margin-bottom: 0; font-size: 1rem; }

.tm-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
.tm-metric-box {
    background: #F1F5F9;
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.tm-metric-box .tm-value { font-size: 1.6rem; font-weight: 800; }
.tm-metric-box .tm-label { font-size: 0.78rem; color: #64748B; text-transform: uppercase; }
.tm-score-high { color: #15803D; }
.tm-score-mid { color: #B45309; }
.tm-score-low { color: #B91C1C; }

.stButton>button[kind="primary"] { background: #4F46E5; color: #fff; font-weight: 700; border-radius: 10px; }
.stButton>button[kind="primary"]:hover { background: #4338CA; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===========================================================================
# TEXT EXTRACTION (PDF & DOCX)
# ===========================================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts)

def extract_resume_text(uploaded_file):
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    try:
        file_bytes = uploaded_file.read()
        if ext == "pdf":
            return extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            return extract_text_from_docx(file_bytes)
    except Exception as exc:
        st.error(f"⚠️ Could not read {uploaded_file.name}: {exc}")
    return None

# ===========================================================================
# GROQ API INTEGRATION (Unified Extraction & Analysis)
# ===========================================================================
def build_unified_prompt(resume_text: str, jd_text: str) -> str:
    return f"""You are an expert HR AI assistant. Analyze the CANDIDATE RESUME against the JOB DESCRIPTION.

CANDIDATE RESUME:
{resume_text[:12000]}

JOB DESCRIPTION:
{jd_text}

Return ONLY a valid JSON object with exactly the following keys. Do not include markdown fences or explanations.
{{
    "name": "Candidate's full name (or 'Unknown')",
    "email": "Candidate's email address (or 'Not Found')",
    "phone": "Candidate's phone number (or 'Not Found')",
    "skills": "Core skills extracted from the resume (comma-separated string)",
    "match_score": A number between 0 and 100 representing how well the resume matches the JD,
    "missing_skills": ["List", "of", "key JD skills", "missing from resume"],
    "custom_questions": ["Question 1 targeting a gap", "Question 2 targeting a gap", "Question 3 targeting a gap"]
}}
"""

def analyze_and_extract_with_groq(client, resume_text: str, jd_text: str, file_name: str):
    try:
        prompt = build_unified_prompt(resume_text, jd_text)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        
        # Safely parse JSON
        raw_content = response.choices[0].message.content.strip()
        result = json.loads(raw_content)
        
        # Ensure all keys exist
        return {
            "file_name": file_name,
            "name": result.get("name", "Unknown"),
            "email": result.get("email", "Not Found"),
            "phone": result.get("phone", "Not Found"),
            "skills": result.get("skills", "Not Found"),
            "match_score": float(result.get("match_score", 0)),
            "missing_skills": result.get("missing_skills", []),
            "custom_questions": result.get("custom_questions", [])
        }
    except Exception as exc:
        st.error(f"⚠️ Groq analysis failed for **{file_name}**: {exc}")
        return None

# ===========================================================================
# EXCEL EXPORT
# ===========================================================================
def build_results_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Match Score (%)": r["match_score"],
            "Candidate Name": r["name"],
            "Email": r["email"],
            "Phone": r["phone"],
            "Extracted Skills": r["skills"],
            "Missing Skills (vs JD)": "; ".join(r.get("missing_skills", [])) or "None",
            "Interview Questions": " | ".join(r.get("custom_questions", [])) or "None",
            "Source File": r["file_name"],
        })
    return pd.DataFrame(rows).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)

def dataframe_to_formatted_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
        workbook = writer.book
        worksheet = writer.sheets["Candidates"]

        header_format = workbook.add_format({
            "bold": True, "bg_color": "#1E293B", "font_color": "#FFFFFF",
            "border": 1, "align": "center", "valign": "vcenter",
        })
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_format)
            width = 25 if col_name not in ["Missing Skills (vs JD)", "Interview Questions", "Extracted Skills"] else 50
            worksheet.set_column(col_idx, col_idx, width, wrap_format)
        worksheet.freeze_panes(1, 0)

    buffer.seek(0)
    return buffer.getvalue()

# ===========================================================================
# UI & APP FLOW
# ===========================================================================
if "results" not in st.session_state:
    st.session_state.results = []

with st.sidebar:
    st.markdown(f"## 🎯 {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.markdown("---")
    
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.success("Groq API key loaded from secrets.")
    else:
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    
    st.markdown("---")
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.results = []
        st.rerun()

st.markdown(f'<div class="tm-header"><h1>🎯 {APP_NAME}</h1><p>{APP_TAGLINE}</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="tm-card"><h4>📋 Job Description</h4>', unsafe_allow_html=True)
    jd_text = st.text_area("Paste JD", height=260, label_visibility="collapsed", placeholder="Paste Job Description here...")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="tm-card"><h4>📥 Upload Resumes</h4>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload resumes", type=ACCEPTED_TYPES, accept_multiple_files=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Process & Screen Candidates", type="primary", use_container_width=True, disabled=not (uploaded_files and jd_text.strip() and groq_api_key)):
    client = Groq(api_key=groq_api_key)
    results = []
    
    progress = st.progress(0.0, text="Initializing...")
    for i, file in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files), text=f"Processing {file.name}...")
        
        text = extract_resume_text(file)
        if text:
            analysis = analyze_and_extract_with_groq(client, text, jd_text, file.name)
            if analysis:
                results.append(analysis)
                
    progress.empty()
    st.session_state.results = results
    st.success(f"Successfully processed {len(results)} candidates!")

# --- DISPLAY RESULTS ---
if st.session_state.results:
    results = st.session_state.results

    st.markdown('<div class="tm-card"><h4>📊 Screening Overview</h4>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Total Candidates Processed", len(results))
    avg_score = round(sum(r["match_score"] for r in results) / len(results), 1)
    m2.metric("Average Match Score", f"{avg_score}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tm-card"><h4>🧾 Candidate Details & Rankings</h4>', unsafe_allow_html=True)
    
    # Sort results by match score
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    
    for rank, cand in enumerate(results, start=1):
        with st.expander(f"#{rank} — {cand['name']} | Score: {cand['match_score']}%", expanded=(rank == 1)):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**📧 Email:** {cand['email']}")
                st.markdown(f"**📱 Phone:** {cand['phone']}")
                st.markdown(f"**🛠️ Extracted Skills:** {cand['skills']}")
                
                score_class = "tm-score-high" if cand["match_score"] >= 75 else ("tm-score-mid" if cand["match_score"] >= 50 else "tm-score-low")
                st.markdown(f'<div class="tm-metric-box" style="margin-top: 15px;"><div class="tm-value {score_class}">{cand["match_score"]}%</div><div class="tm-label">Match Score</div></div>', unsafe_allow_html=True)

            with c2:
                st.markdown("**❌ Missing Skills (vs. JD):**")
                if cand["missing_skills"]:
                    for skill in cand["missing_skills"]:
                        st.markdown(f"- {skill}")
                else:
                    st.caption("No significant gaps identified.")
                
                st.markdown("**🗣️ Suggested Interview Questions:**")
                for q in cand["custom_questions"]:
                    st.markdown(f"- {q}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tm-card"><h4>⬇️ Export Master Sheet</h4>', unsafe_allow_html=True)
    df = build_results_dataframe(results)
    st.download_button(
        "Download Formatted Excel (.xlsx)",
        data=dataframe_to_formatted_excel_bytes(df),
        file_name="Super_TalentMatch_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
