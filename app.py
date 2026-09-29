"""
HireMatrix AI — Universal Enterprise ATS & Resume Screener
=============================================================================
Commercial B2B Edition for Recruitment Agencies & Corporate HR Teams.
Features: 
- Permanent Master Admin Profile (Executive Admin)
- 1-Click Guest Testing Profiles (Name + PIN only, No OTP)
- Auto-Wipe Guest Profiles & Test Data on Logout / Lock Portal
- Bulk CV Extraction, Groq LLM JD Matching & Formatted Excel Export
"""

import io
import json
import os
import re
import hashlib
import random
from datetime import datetime
import pandas as pd
import streamlit as st
from groq import Groq
from supabase import create_client, Client
from PIL import Image, ImageDraw

APP_NAME = "HireMatrix AI"
APP_TAGLINE = "Next-Gen Talent Intelligence & Automated Candidate Screening Portal"
GROQ_MODEL = "openai/gpt-oss-120b"
ACCEPTED_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]

PERMANENT_ADMIN_EMAIL = "admin@hirematrix.ai"
PERMANENT_ADMIN_NAME = "Executive Admin"
PERMANENT_ADMIN_PIN = "1234"

def get_app_favicon():
    img = Image.new("RGBA", (64, 64), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    draw.polygon([(32, 8), (56, 32), (32, 56), (8, 32)], outline=(6, 182, 212), width=4)
    draw.polygon([(32, 20), (44, 32), (32, 44), (20, 32)], fill=(6, 182, 212))
    return img

st.set_page_config(
    page_title=f"{APP_NAME} | Enterprise ATS",
    page_icon=get_app_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "profile" in st.query_params:
    selected_prof = st.query_params["profile"]
    if isinstance(selected_prof, list):
        selected_prof = selected_prof[0]
    del st.query_params["profile"]
    if not st.session_state.get("selected_profile_email"):
        st.session_state.selected_profile_email = selected_prof

@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"⚠️ Supabase Connection Error: {e}")
        return None

supabase: Client = init_supabase()

def ensure_permanent_admin_exists():
    if not supabase:
        return
    try:
        res = supabase.table("hr_users").select("email").eq("email", PERMANENT_ADMIN_EMAIL).execute()
        if not res.data or len(res.data) == 0:
            supabase.table("hr_users").insert({
                "email": PERMANENT_ADMIN_EMAIL,
                "name": PERMANENT_ADMIN_NAME,
                "pin": PERMANENT_ADMIN_PIN,
                "role": "Admin",
                "is_verified": 1
            }).execute()
        else:
            supabase.table("hr_users").update({"name": PERMANENT_ADMIN_NAME}).eq("email", PERMANENT_ADMIN_EMAIL).execute()
    except Exception:
        pass

ensure_permanent_admin_exists()

def get_all_verified_profiles():
    if not supabase:
        return [(PERMANENT_ADMIN_EMAIL, PERMANENT_ADMIN_NAME, PERMANENT_ADMIN_PIN, "Admin")]
    profiles = []
    try:
        res = supabase.table("hr_users").select("*").order("id", desc=False).execute()
        if res.data:
            has_admin = False
            for r in res.data:
                p_email = r.get("email")
                p_name = r.get("name")
                p_pin = str(r.get("pin", ""))
                p_role = r.get("role", "Recruiter")
                if p_email == PERMANENT_ADMIN_EMAIL:
                    has_admin = True
                    p_name = PERMANENT_ADMIN_NAME
                profiles.append((p_email, p_name, p_pin, p_role))
            if not has_admin:
                profiles.insert(0, (PERMANENT_ADMIN_EMAIL, PERMANENT_ADMIN_NAME, PERMANENT_ADMIN_PIN, "Admin"))
    except Exception:
        profiles = [(PERMANENT_ADMIN_EMAIL, PERMANENT_ADMIN_NAME, PERMANENT_ADMIN_PIN, "Admin")]
    return profiles

def verify_user_pin(email, entered_pin):
    clean_email = email.lower().strip()
    entered_str = str(entered_pin).strip()
    
    if clean_email == PERMANENT_ADMIN_EMAIL.lower() and entered_str == PERMANENT_ADMIN_PIN:
        return True, PERMANENT_ADMIN_NAME, "Admin", False

    if not supabase:
        return False, None, None, False

    try:
        res = supabase.table("hr_users").select("*").ilike("email", clean_email).execute()
        if res.data and len(res.data) > 0:
            rec = res.data[0]
            if str(rec.get("pin", "")).strip() == entered_str:
                role = rec.get("role", "Guest Recruiter")
                is_guest = (clean_email != PERMANENT_ADMIN_EMAIL.lower())
                return True, rec.get("name", "Tester"), role, is_guest
    except Exception:
        pass
    return False, None, None, False

def create_guest_tester_profile(name, pin):
    clean_name = name.strip()
    clean_pin = str(pin).strip()
    if not supabase:
        return False, "Database connection not available", None
    
    rand_tag = random.randint(1000, 9999)
    guest_email = f"guest_{rand_tag}@tester.ai"
    
    try:
        data = {
            "email": guest_email,
            "name": clean_name,
            "pin": clean_pin,
            "role": "Guest Recruiter",
            "is_verified": 1
        }
        supabase.table("hr_users").insert(data).execute()
        return True, "Guest profile created!", guest_email
    except Exception as e:
        return False, f"Error: {e}", None

def wipe_guest_session_data(guest_email):
    """Guest tester ke logout hone par uski profile aur test data delete karta hai"""
    if not supabase or not guest_email or guest_email == PERMANENT_ADMIN_EMAIL:
        return
    try:
        supabase.table("hr_users").delete().eq("email", guest_email).execute()
        supabase.table("candidates").delete().eq("email", guest_email).execute()
    except Exception:
        pass

# ----------------- REPOSITORY & DATABASE -----------------
def load_database():
    expected_cols = ["Name", "Father Name", "Qualification", "CGPA", "Passing Year", "Institute", "DOB", "Email", "Phone Number", "Experience", "Latest Experience", "Reference"]
    if not supabase: return pd.DataFrame(columns=expected_cols)
    try:
        res = supabase.table("candidates").select("*").order("id", desc=False).execute()
        if res.data:
            return pd.DataFrame([{
                "Name": r.get("candidate_name") or r.get("name", "Unknown"),
                "Father Name": r.get("father_name", "Not Provided"),
                "Qualification": r.get("education", "Not Provided"),
                "CGPA": r.get("cgpa", "Not Provided"),
                "Passing Year": r.get("passing_year", "Not Provided"),
                "Institute": r.get("university_name", "Not Provided"),
                "DOB": r.get("dob", "Not Provided"),
                "Email": r.get("email", "Not Provided"),
                "Phone Number": r.get("phone", "Not Provided"),
                "Experience": str(r.get("experience_years", "0")),
                "Latest Experience": r.get("latest_experience", "Not Provided"),
                "Reference": r.get("reference", "Not Provided")
            } for r in res.data])
    except Exception: pass
    return pd.DataFrame(columns=expected_cols)

def save_candidates_to_repository(new_candidates):
    if not supabase: return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in new_candidates:
        payload = {
            "candidate_name": c.get("name", "Unknown"),
            "father_name": c.get("father_name", "Not Provided"),
            "education": c.get("education", "Not Provided"),
            "cgpa": c.get("cgpa", "Not Provided"),
            "passing_year": c.get("passing_year", "Not Provided"),
            "university_name": c.get("university_name", "Not Provided"),
            "dob": c.get("dob", "Not Provided"),
            "email": c.get("email", "Not Provided"),
            "phone": c.get("phone", "Not Provided"),
            "experience_years": str(c.get("experience_years", "0")),
            "latest_experience": c.get("latest_experience", "Not Provided"),
            "reference": c.get("reference", "Not Provided"),
            "pipeline_status": "Talent Pool",
            "added_at": timestamp
        }
        try: supabase.table("candidates").insert(payload).execute()
        except Exception: pass

def clear_candidate_database():
    if supabase:
        try: supabase.table("candidates").delete().neq("id", 0).execute()
        except Exception: pass

def save_screened_to_supabase(screened_list):
    if not supabase or not screened_list: return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in screened_list:
        skills_str = ", ".join(r.get("missing_skills", [])) if isinstance(r.get("missing_skills"), list) else str(r.get("missing_skills", ""))
        payload = {
            "job_title": r.get("job_title", "Not Specified"),
            "candidate_name": r.get("name", "Unknown"),
            "father_name": r.get("father_name", "Not Provided"),
            "education": r.get("education", "Not Provided"),
            "cgpa": r.get("cgpa", "Not Provided"),
            "passing_year": r.get("passing_year", "Not Provided"),
            "university_name": r.get("university_name", "Not Provided"),
            "dob": r.get("dob", "Not Provided"),
            "email": r.get("email", "Not Provided"),
            "phone": r.get("phone", "Not Provided"),
            "experience_years": str(r.get("experience_years", "0")),
            "latest_experience": r.get("latest_experience", "Not Provided"),
            "reference": r.get("reference", "Not Provided"),
            "match_score": float(r.get("match_score", 0)),
            "missing_skills": skills_str,
            "pipeline_status": r.get("pipeline_status", "Shortlisted"),
            "screened_at": timestamp
        }
        try: supabase.table("screened_candidates").insert(payload).execute()
        except Exception: pass

def generate_excel(df: pd.DataFrame, sheet_name="Data") -> bytes:
    import openpyxl
    buffer = io.BytesIO()
    export_df = df.copy()
    export_df.insert(0, "Sr. No.", range(1, len(export_df) + 1))
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------- MODERN B2B SAAS CSS -----------------
MODERN_SAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
[data-testid="stSidebar"] { display: none !important; }

.cyber-header-box { text-align: center; padding: 1.8rem 1rem 1.2rem 1rem; margin-bottom: 1.2rem; }
.cyber-title { font-size: 3.2rem !important; font-weight: 800 !important; color: #FFFFFF !important; margin: 0 0 8px 0 !important; }
.cyber-title-pro { color: #06B6D4 !important; background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.cyber-badge { display: inline-flex !important; gap: 8px; background: rgba(6, 182, 212, 0.12) !important; border: 1.5px solid #06B6D4 !important; padding: 5px 20px !important; border-radius: 30px !important; font-size: 0.78rem !important; font-weight: 800 !important; color: #06B6D4 !important; }

.arl-clickable-badge { text-decoration: none !important; color: inherit !important; display: block !important; cursor: pointer !important; }
.cyber-badge-card { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important; border: 1.5px solid #334155 !important; border-left: 5px solid #06B6D4 !important; border-radius: 16px !important; padding: 1.4rem 1.6rem !important; box-shadow: 0 10px 25px rgba(0,0,0,0.4) !important; transition: all 0.25s ease-in-out !important; }
.arl-clickable-badge:hover .cyber-badge-card { border-color: #06B6D4 !important; transform: translateY(-3px) scale(1.01); }

.cyber-top-bar { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255, 255, 255, 0.1); padding-bottom: 0.5rem; margin-bottom: 0.8rem; }
.cyber-access-id { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8; }
.cyber-status-dot { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #06B6D4; font-weight: 700; }
.cyber-avatar-ring { width: 54px; height: 54px; border-radius: 50%; background: #0F172A; border: 2px solid #06B6D4; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; }
.cyber-name-title { margin: 0; font-size: 1.35rem; font-weight: 800; color: #F8FAFC !important; }
.cyber-role-pill { background: rgba(6, 182, 212, 0.15); border: 1px solid #06B6D4; color: #06B6D4; padding: 2px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }

.stButton > button { background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important; color: #FFFFFF !important; border: 1.5px solid #38BDF8 !important; border-radius: 12px !important; font-weight: 700 !important; }
.top-navbar { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1.5px solid #334155; border-bottom: 2px solid #06B6D4; border-radius: 16px; padding: 1.1rem 2rem; margin-bottom: 1.8rem; display: flex; justify-content: space-between; align-items: center; }
.corp-hero { background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1.5px solid #334155; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; border-left: 6px solid #06B6D4; }
.corp-card { background: var(--background-color); border: 1.5px solid rgba(6, 182, 212, 0.25); border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem; }
.metric-box .val { font-size: 1.8rem; font-weight: 800; color: #06B6D4; }
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:first-child { border: 1.5px solid rgba(6, 182, 212, 0.35) !important; border-radius: 24px !important; padding: 2.2rem !important; background: rgba(15, 23, 42, 0.7) !important; }
</style>
"""
st.markdown(MODERN_SAAS_CSS, unsafe_allow_html=True)

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "hr_name" not in st.session_state: st.session_state.hr_name = ""
if "hr_email" not in st.session_state: st.session_state.hr_email = ""
if "hr_role" not in st.session_state: st.session_state.hr_role = "Recruiter"
if "is_guest" not in st.session_state: st.session_state.is_guest = False
if "selected_profile_email" not in st.session_state: st.session_state.selected_profile_email = None
if "screening_results" not in st.session_state: st.session_state.screening_results = []

# ===========================================================================
# LOGIN & INSTANT PROFILE FLOW
# ===========================================================================
if not st.session_state.logged_in:
    saved_profiles = get_all_verified_profiles()
    _, col_c2, _ = st.columns([1, 3.8, 1])
    with col_c2:
        with st.container(border=True):
            st.markdown(f"""
                <div class="cyber-header-box">
                    <h1 class="cyber-title">HireMatrix <span class="cyber-title-pro">AI</span></h1>
                    <div class="cyber-badge"><span>◈</span> {APP_TAGLINE} <span>◈</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            if not st.session_state.selected_profile_email:
                st.markdown("### 👥 Active Recruiter Profiles")
                st.caption("Click on your card to enter your security PIN:")
                
                for p_email, p_name, _, p_role in saved_profiles:
                    badge_label = "OFFICIAL ADMIN" if p_email == PERMANENT_ADMIN_EMAIL else "GUEST TESTER"
                    st.markdown(f"""
                        <a href="?profile={p_email}" target="_self" class="arl-clickable-badge">
                            <div class="cyber-badge-card">
                                <div class="cyber-top-bar">
                                    <span class="cyber-access-id">{badge_label}</span>
                                    <span class="cyber-status-dot">CLICK TO ENTER ➔</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 16px;">
                                    <div class="cyber-avatar-ring">👤</div>
                                    <div>
                                        <h3 class="cyber-name-title">{p_name}</h3>
                                        <span class="cyber-role-pill">{p_role}</span>
                                        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">✉ {p_email}</div>
                                    </div>
                                </div>
                            </div>
                        </a>
                    """, unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                if st.button("➕ Create Instant Test Profile", use_container_width=True):
                    st.session_state.selected_profile_email = "new_guest"
                    st.rerun()

            elif st.session_state.selected_profile_email == "new_guest":
                st.markdown("### 📝 Create Instant Tester Profile")
                st.caption("No email or OTP required. Profile auto-deletes when you lock the portal.")
                with st.form("guest_reg_form"):
                    g_name = st.text_input("Your Name / Company Name", placeholder="e.g. Talent Lead (Agency)")
                    g_pin = st.text_input("Set Any 4-Digit PIN", type="password", max_chars=4, placeholder="••••")
                    create_sub = st.form_submit_button("Enter Portal Immediately", use_container_width=True)
                
                c_b1, c_b2 = st.columns(2)
                if create_sub:
                    if not g_name.strip() or len(g_pin.strip()) != 4 or not g_pin.isdigit():
                        st.warning("Please provide your name and an exact 4-digit numeric PIN.")
                    else:
                        ok, msg, created_email = create_guest_tester_profile(g_name, g_pin)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.hr_name = g_name.strip()
                            st.session_state.hr_email = created_email
                            st.session_state.hr_role = "Guest Recruiter"
                            st.session_state.is_guest = True
                            st.session_state.selected_profile_email = None
                            st.rerun()
                        else:
                            st.error(msg)
                with c_b2:
                    if st.button("Back to Profiles", use_container_width=True):
                        st.session_state.selected_profile_email = None
                        st.rerun()

            else:
                target_email = st.session_state.selected_profile_email
                p_match = next((p for p in saved_profiles if p[0].lower() == target_email.lower()), (target_email, "Recruiter", "", "Recruiter"))
                st.markdown(f"### 🔐 Enter Security PIN: {p_match[1]}")
                st.caption(f"PIN authentication for **{target_email}**")
                
                with st.form("pin_form"):
                    pin = st.text_input("4-Digit PIN", type="password", max_chars=4, placeholder="••••")
                    sub = st.form_submit_button("Access Portal", use_container_width=True)
                c_l1, c_l2 = st.columns(2)
                if sub:
                    ok, name, role, is_guest = verify_user_pin(target_email, pin)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.hr_name = name
                        st.session_state.hr_email = target_email
                        st.session_state.hr_role = role
                        st.session_state.is_guest = is_guest
                        st.session_state.selected_profile_email = None
                        st.rerun()
                    else:
                        st.error("❌ Incorrect PIN. Please try again.")
                with c_l2:
                    if st.button("Switch Profile", use_container_width=True):
                        st.session_state.selected_profile_email = None
                        st.rerun()
    st.stop()

# ===========================================================================
# MAIN RECRUITER PORTAL
# ===========================================================================
df_all = load_database()

col_n1, col_n2 = st.columns([8.2, 1.8], vertical_alignment="center")
with col_n1:
    user_badge = "👑 Official Admin" if not st.session_state.is_guest else "🧪 Guest Session (Auto-Wipe on Lock)"
    st.markdown(f"""
        <div class="top-navbar">
            <div>
                <h2 style="margin: 0; color: #FFFFFF; font-size: 1.55rem;">HireMatrix <span style="color: #06B6D4;">AI</span></h2>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.8rem;">Active: <b>{st.session_state.hr_name}</b> ({st.session_state.hr_email}) &bull; Status: <span style="color: #06B6D4;"><b>{user_badge}</b></span></p>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col_n2:
    if st.button("🚪 Lock Portal", use_container_width=True):
        if st.session_state.is_guest:
            wipe_guest_session_data(st.session_state.hr_email)
        st.session_state.logged_in = False
        st.session_state.hr_name = ""
        st.session_state.hr_email = ""
        st.session_state.hr_role = "Recruiter"
        st.session_state.is_guest = False
        st.session_state.selected_profile_email = None
        st.session_state.screening_results = []
        st.rerun()

st.markdown(f"""
    <div class="corp-hero">
        <h1 style="color: #FFFFFF; margin: 0 0 8px 0;">Automated AI Recruitment Pipeline</h1>
        <p style="color: #CBD5E1; margin: 0;">Welcome, <b>{st.session_state.hr_name}</b> &mdash; Ingest bulk CVs, match against any Job Description, and view instant candidate gap analysis.</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- OCR & EXTRACTION FUNCTIONS -----------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber, pytesseract
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if len(txt.strip()) > 40:
                parts.append(f"\n--- [PAGE {idx}] ---\n" + txt)
            else:
                try:
                    img = page.to_image(resolution=150).original.convert("L")
                    ocr_t = pytesseract.image_to_string(img)
                    if ocr_t.strip(): parts.append(f"\n--- [PAGE {idx} (OCR)] ---\n" + ocr_t)
                except Exception: pass
    return "\n\n".join(parts)

def extract_candidates(client, resume_text: str, file_name: str):
    prompt = f"""Extract all candidate resumes from this document text.
Return ONLY valid JSON matching schema:
{{"candidates": [{{"name": "Full Name", "father_name": "Father Name or Not Provided", "education": "Degree Title", "cgpa": "CGPA", "passing_year": "Year", "university_name": "Institute", "dob": "DOB", "email": "Email", "phone": "Phone", "experience_years": "Years", "latest_experience": "Role", "reference": "Reference", "skills": "Skills"}}]}}
DOCUMENT TEXT:\n{resume_text[:28000]}"""
    try:
        res = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.1)
        parsed = json.loads(res.choices[0].message.content.strip())
        return parsed.get("candidates", []) if isinstance(parsed, dict) else parsed
    except Exception: return []

def evaluate_jd(client, cand_row, jd_text):
    prompt = f"""Evaluate candidate against Job Description.
CANDIDATE: {cand_row['Name']}, {cand_row['Qualification']}, Exp: {cand_row['Experience']}
JD: {jd_text}
Return ONLY JSON: {{"match_score": 0-100, "is_relevant": true/false, "missing_skills": ["List"]}}"""
    try:
        res = client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.2)
        r = json.loads(res.choices[0].message.content.strip())
        return float(r.get("match_score", 0)), bool(r.get("is_relevant", True)), r.get("missing_skills", [])
    except Exception: return 0.0, True, []

# ----------------- TABS WORKFLOW -----------------
tab1, tab2, tab3 = st.tabs(["📥 1. Ingest Resumes", "🎯 2. Screen & Match", "🗄️ 3. Master Talent Grid"])

with tab1:
    st.markdown('<div class="corp-card"><h4>📥 Bulk Ingestion (PDF / DOCX)</h4>', unsafe_allow_html=True)
    files = st.file_uploader("Upload resumes", type=ACCEPTED_TYPES, accept_multiple_files=True)
    g_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    
    if st.button("⚡ Process & Extract Resumes", type="primary", use_container_width=True, disabled=not (files and g_key)):
        client = Groq(api_key=g_key)
        extracted = []
        progress = st.progress(0.0, text="Extracting candidate profiles via Groq...")
        for i, f in enumerate(files):
            progress.progress((i + 1) / (len(files) + 1), text=f"Processing {f.name}...")
            txt = extract_text_from_pdf(f.read()) if f.name.endswith(".pdf") else ""
            if txt: extracted.extend(extract_candidates(client, txt, f.name))
        progress.empty()
        if extracted:
            save_candidates_to_repository(extracted)
            st.success(f"🎉 Successfully extracted and saved {len(extracted)} candidate(s)!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="corp-card"><h4>🎯 AI Screening against Job Description</h4>', unsafe_allow_html=True)
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        target_role = st.text_input("Target Position Title", value="Senior Software Engineer")
    with col_j2:
        thresh = st.slider("Minimum Match Score Highlight (%)", 0, 100, 50)
    
    jd = st.text_area("Paste Full Job Description Requirements", height=120, placeholder="Requirements, skills, experience...")
    
    if st.button("⚡ Run Deep AI Screening", type="primary", use_container_width=True, disabled=not (jd and not df_all.empty and g_key)):
        client = Groq(api_key=g_key)
        screened = []
        progress = st.progress(0.0, text="Evaluating candidates...")
        for idx, (_, row) in enumerate(df_all.iterrows()):
            progress.progress((idx + 1) / (len(df_all) + 1))
            sc, rel, miss = evaluate_jd(client, row, jd)
            if rel:
                screened.append({
                    **row.to_dict(),
                    "job_title": target_role,
                    "match_score": sc,
                    "missing_skills": miss,
                    "pipeline_status": "Shortlisted" if sc >= thresh else "Talent Pool"
                })
        progress.empty()
        screened.sort(key=lambda x: x["match_score"], reverse=True)
        st.session_state.screening_results = screened
        save_screened_to_supabase(screened)
        st.success(f"Screening complete! {len(screened)} candidate(s) evaluated.")
        st.rerun()

    results = st.session_state.get("screening_results", [])
    if results:
        st.download_button("📊 Export Screened Excel", generate_excel(pd.DataFrame(results)), "Screened_Candidates.xlsx", use_container_width=True)
        for idx, cand in enumerate(results, start=1):
            with st.expander(f"#{idx} — {cand['Name']} | Match: {cand['match_score']}% | Stage: {cand['pipeline_status']}"):
                st.write(f"🎓 **Education:** {cand['Qualification']} ({cand['Institute']}) | 💼 **Experience:** {cand['Experience']}")
                st.write(f"❌ **Missing Skills vs JD:** {', '.join(cand['missing_skills']) if cand['missing_skills'] else 'None'}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="corp-card"><h4>🗄️ Master Talent Repository</h4>', unsafe_allow_html=True)
    df_live = load_database()
    if not df_live.empty:
        grid_df = df_live.copy()
        grid_df.insert(0, "Sr. No.", range(1, len(grid_df) + 1))
        st.dataframe(grid_df, use_container_width=True)
        st.download_button("📊 Download Master Excel", generate_excel(df_live), "Master_Talent_Pool.xlsx", use_container_width=True)
        if st.button("🗑️ Clear Talent Pool"):
            clear_candidate_database()
            st.rerun()
    else:
        st.info("No candidates in repository yet. Upload resumes in Step 1.")
    st.markdown("</div>", unsafe_allow_html=True)
