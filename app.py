import streamlit as st
import joblib
import torch
import pandas as pd
import pdfplumber
from docx import Document
import re
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Requirement AI Classifier Pro (Optimized)", layout="wide")
st.title("🛡️ Software Requirement AI Batch Processor (Optimized Models)")

# ==========================================
# 2. CATEGORY MAPPING (The Legend)
# ==========================================
ABBREVIATION_MAP = {
    'F': 'Functional', 'A': 'Availability', 'L': 'Legal', 'LF': 'Look & Feel',
    'MN': 'Maintainability', 'O': 'Operational', 'PE': 'Performance',
    'SC': 'Scalability', 'SE': 'Security', 'US': 'Usability', 'PO': 'Portability',
    'FT': 'Fault Tolerance',
    'Notification': 'Notification (Functional)', 'User_Profile': 'User Profile (Functional)',
    'Report_Output': 'Report/Output (Functional)', 'Data_Input': 'Data Input (Functional)',
    'General_Functional': 'General Functional'
}

st.sidebar.header("📖 Category Legend")
for key, value in ABBREVIATION_MAP.items():
    st.sidebar.markdown(f"**{key}**: {value}")

st.sidebar.divider()
st.sidebar.header("⚙️ Mitigation Strategies Used")
st.sidebar.markdown("""
- **SMOTE Balancing** — improved SVM robustness on rare classes (Iteration 2).
- **Weighted Cross-Entropy** — improved DistilBERT minority-class detection (Iteration 2).
- **Functional Sub-typing** — the 'F' class is split into 5 sub-classes.
""")

# ==========================================
# 3. MODEL PATHS — SVM ships in the repo, BERT loads from Hugging Face Hub
# ==========================================
SVM_PATH = "svm_model_optimized.pkl"   # small file, committed directly to GitHub
BERT_REPO = "Tendou012/distillbert-requirement-classifier"  # confirmed HF repo
BERT_MAX_LENGTH = 128                  # matches the weighted model's training tokenization

# ------------------------------------------------------------------
# Check the SVM file actually exists before trying to load it.
# The BERT model no longer lives in this repo — it's pulled from
# Hugging Face Hub at startup, so there's no local folder to check
# for that one. If BERT_REPO is wrong or private, load_bert() below
# will raise a clear error instead.
# ------------------------------------------------------------------
missing = []
if not os.path.isfile(SVM_PATH):
    missing.append(f"'{SVM_PATH}' file")

if missing:
    st.error(
        "❌ Missing model file(s) in the app's working directory:\n\n- "
        + "\n- ".join(missing)
        + "\n\nMake sure these are uploaded/committed alongside app.py."
    )
    st.stop()


@st.cache_resource
def load_svm():
    return joblib.load(SVM_PATH)

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO)
    return tokenizer, model

try:
    svm_model = load_svm()
except Exception as e:
    st.error(f"Error loading SVM model: {e}")
    st.stop()

try:
    bert_tokenizer, bert_model = load_bert()
except Exception as e:
    st.error(
        f"Error loading BERT model from Hugging Face Hub ('{BERT_REPO}'): {e}\n\n"
        "Common causes: (1) the repo ID is misspelled, (2) the repo is set to "
        "Private instead of Public, or (3) the uploaded files aren't sitting "
        "at the repo root (config.json, model weights, and tokenizer files "
        "must be directly in the repo, not inside a subfolder)."
    )
    st.stop()

# Defensive fix: id2label keys can come back as strings after a
# save/reload round-trip through JSON. Coerce to int once, up front.
bert_id2label = {int(k): v for k, v in bert_model.config.id2label.items()}

# ==========================================
# 4. PROCESSING UTILITIES
# ==========================================
def split_into_sentences(text):
    if not text: return []
    sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
    return [s.strip() for s in sentences if len(s.strip()) > 12]

def get_prediction(text, model_type):
    if not text or str(text).strip() == "nan": return "Empty Content"
    if model_type == "SVM":
        pred_code = svm_model.predict([str(text)])[0]
    else:
        inputs = bert_tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=BERT_MAX_LENGTH)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        pred_code = bert_id2label[pred_idx]
    return ABBREVIATION_MAP.get(pred_code, f"Unknown ({pred_code})")

# ==========================================
# 5. INPUT INTERFACE
# ==========================================
st.sidebar.divider()
model_choice = st.sidebar.radio("Select Classification Engine:", ("SVM", "BERT"))

tab1, tab2 = st.tabs(["📄 Upload Document", "✍️ Paste Text"])

# Logic to hold data
if 'temp_df' not in st.session_state:
    st.session_state['temp_df'] = None

with tab1:
    uploaded_file = st.file_uploader("Upload PDF, DOCX, TXT, CSV, or XLSX", type=["csv", "xlsx", "pdf", "docx", "txt"])
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_type == 'pdf':
                from pdfplumber import open as open_pdf
                with open_pdf(uploaded_file) as pdf:
                    text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                st.session_state['temp_df'] = pd.DataFrame(split_into_sentences(text), columns=['Requirement'])
            elif file_type == 'docx':
                doc = Document(uploaded_file)
                text = " ".join([p.text for p in doc.paragraphs])
                st.session_state['temp_df'] = pd.DataFrame(split_into_sentences(text), columns=['Requirement'])
            elif file_type == 'txt':
                content = uploaded_file.read().decode("utf-8")
                st.session_state['temp_df'] = pd.DataFrame(split_into_sentences(content), columns=['Requirement'])
            elif file_type == 'csv':
                try: df = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                st.session_state['temp_df'] = df
            else:
                st.session_state['temp_df'] = pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    raw_paste = st.text_area("Paste text here:", height=150)
    if st.button("Process Pasted Text"):
        st.session_state['temp_df'] = pd.DataFrame(split_into_sentences(raw_paste), columns=['Requirement'])

# ==========================================
# 6. DYNAMIC COLUMN SELECTION & ANALYSIS
# ==========================================
if st.session_state['temp_df'] is not None:
    df = st.session_state['temp_df']
    st.divider()
    st.subheader("🔍 Data Preview & Column Selection")
    st.write("Review your data below and select which column the AI should read.")
    st.dataframe(df.head(5), use_container_width=True)

    col_to_analyze = st.selectbox("Which column contains the requirements/sentences?", df.columns)

    if st.button("🚀 Start AI Analysis"):
        with st.spinner("Analyzing..."):
            df['Predicted_Category'] = df[col_to_analyze].apply(lambda x: get_prediction(x, model_choice))
            df['Type'] = df['Predicted_Category'].apply(lambda x: "Functional (F)" if "Functional" in x else "Non-Functional (NF)")
            st.session_state['result_df'] = df
            st.session_state['final_col'] = col_to_analyze

# ==========================================
# 7. DASHBOARD (Remains stable with Session State)
# ==========================================
if 'result_df' in st.session_state:
    res = st.session_state['result_df']
    f_col = st.session_state['final_col']

    st.divider()
    st.header("📊 Classification Dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total items", len(res))
    m2.metric("Functional", len(res[res['Type'] == "Functional (F)"]))
    m3.metric("Non-Functional", len(res[res['Type'] == "Non-Functional (NF)"]))

    st.subheader("📋 Summary Table")
    st.table(res.groupby(['Type', 'Predicted_Category']).size().reset_index(name='Count'))

    st.subheader("🎯 Filtered Results")
    choice = st.selectbox("Filter Category:", ["All"] + sorted(res['Predicted_Category'].unique()))
    disp = res if choice == "All" else res[res['Predicted_Category'] == choice]
    st.dataframe(disp[[f_col, 'Predicted_Category', 'Type']], use_container_width=True)

    st.download_button("📥 Download Result", res.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
