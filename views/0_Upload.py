"""
Streamlit View: 0_Upload.py
Upload and Ingest Patient Discharge Summaries & Clinical Claims with Automated Readmission Scoring.
"""
import os
import time
import streamlit as st
from datetime import datetime
from src.extraction.extraction_service import extract_text
from src.extraction.metadata_service import extract_metadata
from src.indexing.duplicate_detection_service import DuplicateDetectionService
from src.ml_services.readmission_predictor import predict_readmission_risk
from src.utils.supabase_client import insert_clinical_record, get_clinical_records, upload_document_to_storage

st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.5rem;
">
    <h1 style="margin: 0; font-family: 'Outfit', sans-serif; color: #0F172A; font-size: 2.2rem; font-weight: 700;">Clinical Document & Claim Ingestion</h1>
    <p style="margin: 6px 0 0 0; color: #475569; font-size: 1.05rem;">Upload patient discharge summaries or hospital invoices for AI parsing, 30-day readmission scoring, and automated adjudication.</p>
</div>
""", unsafe_allow_html=True)

records = get_clinical_records(limit=100)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Ingested Records", len(records))
with c2:
    high_risk_n = len([r for r in records if float(r.get("readmission_risk", 0.0) or 0.0) >= 0.60])
    st.metric("High Readmission Risk", high_risk_n)
with c3:
    pending_n = len([r for r in records if r.get("adjudication_status") == "Pending Review"])
    st.metric("Pending Adjudication", pending_n)
with c4:
    st.metric("Database Storage", "Supabase Cloud", delta="PostgreSQL Active")

st.markdown("---")

st.markdown("### Upload Discharge Summaries & Clinical Notes")
uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, or Image files",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process & Ingest Clinical Documents", type="primary", use_container_width=True):
        dedupe_service = DuplicateDetectionService()
        
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            with st.status(f"Processing: {uploaded_file.name}", expanded=True) as status_box:
                st.write("Running Layer 1 Exact Duplicate Check...")
                if dedupe_service.is_exact_duplicate(file_bytes):
                    st.warning(f"Exact duplicate document detected for {uploaded_file.name}. Skipped.")
                    status_box.update(label=f"Skipped (Duplicate): {uploaded_file.name}", state="complete", expanded=False)
                    continue

                st.write("Extracting clinical text (Digital & OCR)...")
                ext_res = extract_text(file_bytes, extension=ext)
                text = ext_res.content
                confidence = ext_res.confidence

                if dedupe_service.is_near_duplicate(text, file_bytes, uploaded_file.name):
                    st.warning(f"Near-duplicate text similarity detected for {uploaded_file.name}.")

                st.write("Extracting clinical diagnoses, comorbidities & SDoH via Gemini...")
                metadata = extract_metadata(text)
                if not metadata or not isinstance(metadata, dict):
                    metadata = {}

                st.write("Computing 30-Day Readmission Probability (Calibrated Risk Engine)...")
                risk_score, feature_impacts = predict_readmission_risk(metadata)
                metadata["readmission_risk"] = risk_score
                
                if risk_score >= 0.60:
                    adj_status = "Flagged for Audit"
                    review_reason = f"High Readmission Risk ({risk_score*100:.1f}%)"
                else:
                    adj_status = "Approved"
                    review_reason = "Standard Low-Risk Clearance"

                st.write("Indexing in Supabase PostgreSQL & generating RAG Embeddings...")
                patient_id = metadata.get("patient_id") or f"PAT-{abs(hash(uploaded_file.name)) % 100000}"
                
                record_payload = {
                    "patient_id": str(patient_id),
                    "file_name": uploaded_file.name,
                    "document_number": metadata.get("document_number") or f"DOC-{int(time.time())%100000}",
                    "entity_name": metadata.get("patient_name") or metadata.get("entity_name") or "Patient",
                    "admission_date": metadata.get("admission_date"),
                    "discharge_date": metadata.get("discharge_date") or metadata.get("document_date"),
                    "length_of_stay": int(metadata.get("length_of_stay", 3) or 3),
                    "primary_diagnosis": metadata.get("primary_diagnosis") or "General Inpatient Care",
                    "comorbidities": metadata.get("comorbidities") or [],
                    "num_medications": int(metadata.get("num_medications", 4) or 4),
                    "follow_up_scheduled": bool(metadata.get("follow_up_scheduled", False)),
                    "lives_alone": bool(metadata.get("lives_alone", False)),
                    "readmission_risk": float(risk_score),
                    "adjudication_status": adj_status,
                    "review_reason": review_reason,
                    "extracted_text": text
                }
                
                # Upload to Supabase Free Storage & Cache for visual rendering
                st.session_state[f"doc_bytes_{uploaded_file.name}"] = file_bytes
                doc_storage_url = upload_document_to_storage(file_bytes, uploaded_file.name)

                created_id = insert_clinical_record(record_payload)
                if created_id:
                    dedupe_service.log_document(file_bytes, created_id, text=text, filename=uploaded_file.name)
                
                status_box.update(label=f"Ingested: {uploaded_file.name} (Risk: {risk_score*100:.1f}%)", state="complete", expanded=False)

                risk_color = "#DC2626" if risk_score >= 0.6 else ("#D97706" if risk_score >= 0.3 else "#16A34A")
                st.markdown(f"""
                <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:15px; border-radius:8px; margin-bottom:15px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <h4 style='margin:0; color:#1E293B;'>Document: {uploaded_file.name} (Patient <code>{patient_id}</code>)</h4>
                        <span style='color:{risk_color}; font-weight:700; font-size:15px;'>{risk_score*100:.1f}% Readmission Risk</span>
                    </div>
                    <p style='margin:6px 0 2px 0;'><b>Primary Diagnosis:</b> {record_payload['primary_diagnosis']}</p>
                    <p style='margin:2px 0;'><b>Adjudication Status:</b> <code>{adj_status}</code></p>
                </div>
                """, unsafe_allow_html=True)
                
        st.success("All documents processed and stored in Supabase.")
        time.sleep(1)
        st.rerun()
