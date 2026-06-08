import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Student Outcome Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 14px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1{font-size: 2rem; margin: 0 0 0.3rem 0;}
    .main-header p{font-size: 0.95rem; opacity: 0.85; margin: 0;}

    .result-graduate {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 2rem; border-radius: 14px; color: white; text-align: center; margin-top: 1.5rem;
    }
    .result-dropout {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        padding: 2rem; border-radius: 14px; color: white; text-align: center; margin-top: 1.5rem;
    }
    .result-title {font-size: 2rem; font-weight: 800; margin: 0;}
    .result-prob {font-size: 3rem; font-weight: 900; margin: 0.4rem 0 0 0;}
    .result-subtitle {font-size: 0.95rem; opacity: 0.85; margin-top: 0.3rem;}

    .section-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 1.2px;
        color: #6b7280; text-transform: uppercase;
        margin: 1.6rem 0 0.6rem 0;
        padding-bottom: 0.3rem; border-bottom: 1px solid #e5e7eb;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: white; font-weight: 700; font-size: 1.05rem;
        border: none; border-radius: 8px; padding: 0.7rem 2rem; width: 100%; margin-top: 1rem;
    }
    .stButton>button:hover { background: #1e40af;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    mdl = joblib.load(os.path.join(base, "random_forest.joblib"))
    meta = joblib.load(os.path.join(base, "model_metadata.joblib"))
    pre = mdl.named_steps['preprocessor']
    ohe = pre.named_transformers_['cat']
    valid = {
        'nationality': [int(x) for x in ohe.categories_[3]],
        'course': [int(x) for x in ohe.categories_[2]],
    }
    return mdl, meta.get("threshold", 0.3), valid

model, THRESHOLD, VALID = load_model()

NATIONALITY = dict(sorted({
    1: "Portugal",
    2: "Jerman",
    6: "Spanyol",
    11: "Italia",
    13: "Belanda",
    14: "Inggris",
    17: "Lithuania",
    21: "Angola",
    22: "Cape Verde",
    24: "Guinea",
    25: "Mozambik",
    26: "Sao Tome & Principe",
    41: "Turki",
    62: "Brazil",
    100: "Rumania",
    101: "Moldova",
    103: "Meksiko",
    105: "Ukraina",
    109: "Rusia",
}.items()))

NATIONALITY_DISPLAY = dict(sorted({
    "Indonesia": 62,
    "Malaysia": 62,
    "Singapura": 62,
    "Thailand": 62,
    "Vietnam": 62,
    "Filipina": 62,
    "India": 62,
    "Pakistan": 62,
    "Bangladesh": 62,
    "Sri Lanka": 62,
    "Nepal": 62,
    "Jepang": 62,
    "Korea Selatan": 62,
    "China": 62,
    "Taiwan": 62,
    "Arab Saudi": 62,
    "Iran": 62,
    "Timor Leste": 62,
    "Myanmar": 62,
    "Kamboja": 62,
    "Portugal": 1,
    "Jerman": 2,
    "Spanyol": 6,
    "Italia": 11,
    "Belanda": 13,
    "Inggris": 14,
    "Lithuania": 17,
    "Rumania": 100,
    "Moldova": 101,
    "Ukraina": 105,
    "Rusia": 109,
    "Brazil": 62,
    "Meksiko": 103,
    "Angola": 21,
    "Cape Verde": 22,
    "Guinea": 24,
    "Mozambik": 25,
    "Sao Tome": 26,
    "Turki": 41,
}.items()))

COURSE_DISPLAY = dict(sorted({
    "Teknik Informatika": 9119,
    "Animasi & Desain Multimedia": 171,
    "Desain Komunikasi Visual": 9070,
    "Manajemen Informatika (sore)": 9991,
    "Manajemen": 9147,
    "Pemasaran & Periklanan": 9670,
    "Pekerjaan Sosial": 9238,
    "Pekerjaan Sosial (sore)": 8014,
    "Keperawatan": 9500,
    "Kesehatan Gigi": 9556,
    "Keperawatan Hewan": 9085,
    "Agronomi": 9003,
    "Teknologi Biofuel": 33,
    "Equinkultura": 9130,
    "Jurnalisme & Komunikasi": 9773,
    "Pendidikan Dasar": 9853,
    "Pariwisata": 9254,
}.items()))

st.markdown("""
<div class="main-header">
  <h1>Student Outcome Predictor</h1>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Data Pribadi</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=None, placeholder="Masukkan usia...")
    gender = st.selectbox("Jenis Kelamin", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
with col2:
    nat_label = st.selectbox("Kewarganegaraan", list(NATIONALITY_DISPLAY.keys()), index=None, placeholder="Pilih kewarganegaraan...")
    nationality_code = NATIONALITY_DISPLAY[nat_label] if nat_label is not None else None
    international = 0 if nationality_code == 1 else 1
 
    course_label = st.selectbox("Program Studi", list(COURSE_DISPLAY.keys()), index=None, placeholder="Pilih program studi...")
    course_code  = COURSE_DISPLAY[course_label] if course_label is not None else None
 

st.markdown('<div class="section-label">Kondisi Ekonomi</div>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    tuition_fees = st.selectbox("Uang Kuliah Lancar?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")
with col4:
    scholarship = st.selectbox("Penerima Beasiswa?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")
with col5:
    debtor = st.selectbox("Memiliki Tunggakan?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")

st.markdown('<div class="section-label">Kinerja Akademik - Semester 1</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    cu1_enrolled = st.number_input("SKS (Sem 1)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
with c2:
    cu1_approved = st.number_input("SKS Lulus (Sem 1)", min_value=0, max_value=30,value=None, placeholder="Masukkan angka...")
with c3:
    cu1_ipk = st.number_input("IPK (Sem 1)", min_value=0.0, max_value=4.0, value=None, placeholder="Masukkan angka...", step=0.01, help="Skala 0.00 – 4.00")

st.markdown('<div class="section-label">Kinerja Akademik – Semester 2</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    cu2_enrolled = st.number_input("SKS (Sem 2)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
with c5:
    cu2_approved = st.number_input("SKS Lulus (Sem 2)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
with c6:
    cu2_ipk = st.number_input("IPK (Sem 2)", min_value=0.0, max_value=4.0, value=None, placeholder="Masukkan angka...", step=0.01, help="Skala 0.00 – 4.00")
predict_btn = st.button("Prediksi Kelulusan")

if predict_btn:
    input_data = pd.DataFrame([{
        "Age at enrollment": age,
        "Gender": gender,
        "Nacionality": nationality_code,
        "International": international,
        "Course": course_code,
        "Tuition fees up to date": tuition_fees,
        "Scholarship holder": scholarship,
        "Debtor": debtor,
        "Admission grade": 130.0,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_ipk * 5.0,
        "Curricular units 1st sem (evaluations)": max(cu1_enrolled, cu1_approved),
        "Curricular units 2nd sem (enrolled)": cu2_enrolled,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_ipk * 5.0,
        "Curricular units 2nd sem (evaluations)": max(cu2_enrolled, cu2_approved),
        "Marital status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Daytime/evening attendance": 1,
        "Previous qualification": 1,
        "Previous qualification (grade)": 130.0,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 4,
        "Father's occupation": 4,
        "Displaced": 0,
        "Educational special needs": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74,
    }])

    expected_cols = list(model.feature_names_in_)
    input_data = input_data[expected_cols]

    try:
        proba = model.predict_proba(input_data)[0]
        prob_graduate = proba[0]
        prob_dropout = proba[1]
        prediction = "graduate" if prob_dropout < THRESHOLD else "dropout"
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        st.stop()

    if prediction == "graduate":
        st.markdown(f"""
        <div class="result-graduate">
          <div class="result-title">DIPREDIKSI LULUS</div>
          <div class="result-prob">{prob_graduate:.0%}</div>
          <div class="result-subtitle">Probabilitas mahasiswa ini menyelesaikan studi</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-dropout">
          <div class="result-title">RISIKO DROPOUT</div>
          <div class="result-prob">{prob_dropout:.0%}</div>
          <div class="result-subtitle">Probabilitas mahasiswa ini tidak menyelesaikan studi — perlu perhatian lebih</div>
        </div>
        """, unsafe_allow_html=True)
