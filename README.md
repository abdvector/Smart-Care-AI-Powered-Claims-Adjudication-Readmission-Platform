# Smart-Care: AI-Powered Healthcare Document Intelligence & Claims Adjudication Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![AI Models](https://img.shields.io/badge/AI%20Engine-Google%20Gemini%202.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Overview

**Smart-Care** is an end-to-end, cloud-native clinical document intelligence and healthcare claims adjudication system. It bridges the operational gap between unstructured clinical documentation (inpatient discharge summaries, physician notes, hospital billing invoices) and automated health economics decision-making.

The platform executes three primary workflows:
1. **Multimodal Clinical Parsing & OCR:** Ingests digital PDFs, Word documents, and low-fidelity scanned medical records using dual-engine fallback (digital extraction + Google Gemini multimodal vision).
2. **Predictive 30-Day Readmission Risk & Econometric Modeling:** Implements a calibrated statistical scoring engine combining the Charlson Comorbidity Index, Inpatient Length of Stay (LOS), Polypharmacy burdens, and Social Determinants of Health (SDoH), supported by patient-level Explainable AI (XAI) feature attribution and cost-minimization cutoff theorems.
3. **Claims Adjudication & RAG Vector Search:** Provides an operational adjudication queue with side-by-side in-app document inspection and evidence-grounded Retrieval-Augmented Generation (RAG) powered by 768-dimensional embeddings and Supabase PostgreSQL `pgvector`.

---

## System Architecture

```
[ Clinical Documents / Scans / PDFs ]
                 |
                 v
   +------------------------------+
   |  Ingestion & Validation      | --> Multi-Format Validation (PDF/DOCX/Images)
   +------------------------------+
                 |
                 v
   +------------------------------+
   |  3-Layer Deduplication       | --> Layer 1: SHA-256 Hash
   |  Engine                      | --> Layer 2: MinHash Jaccard Similarity
   +------------------------------+ --> Layer 3: Perceptual Hash (pHash)
                 |
                 v
   +------------------------------+
   |  Multimodal OCR & Extraction | --> Primary: Lightweight Digital Parsing (pypdf)
   |  Service                     | --> Fallback: Google Gemini 2.5 Flash Multimodal Vision
   +------------------------------+
                 |
                 v
   +------------------------------+
   |  Clinical Entity Extraction  | --> Diagnosis, Comorbidities, Procedures, SDoH Factors
   +------------------------------+
                 |
        +--------+--------+
        |                 |
        v                 v
+------------------+ +------------------------------------+
| ML Risk Engine   | | Supabase Cloud Database & Storage  |
| 30-Day Readmit   | | - Object Storage: 'clinical-docs'  |
| XAI Attribution  | | - pgvector Embeddings (768-dim)    |
| Econometric Cut  | | - Structured Clinical Records DB   |
+------------------+ +------------------------------------+
        |                 |
        +--------+--------+
                 |
                 v
+--------------------------------------------------------+
| Streamlit Enterprise Clinical Workspace                |
| - Ingestion Pipeline (0_Upload.py)                     |
| - Vector Search & RAG Chat (1_Search.py)               |
| - Side-by-Side Claims Adjudication (2_Action_Centre.py)|
| - ML Risk Simulator & Economic Curves (5_ML_Predict)   |
+--------------------------------------------------------+
```

---

## Core Capabilities

### 1. Multimodal Document Parsing & OCR Confidence Scoring
* Dual-engine extraction pipeline: handles native text layers via `pypdf` and routes unstructured, low-resolution, or handwritten medical charts to Google Gemini Vision.
* Weighted arithmetic confidence scoring: computes token-level confidence scores with custom medical terminology weighting multipliers.

### 2. Explainable 30-Day Readmission Risk Engine
* **Multivariate Clinical Scoring:** Combines Charlson Comorbidity Index log-odds, non-linear Length of Stay multipliers, polypharmacy medication thresholds, and critical SDoH indicators (lives alone, lack of home caregiver support, scheduled 7-day post-discharge outpatient follow-up).
* **Explainable AI (XAI):** Generates patient-level log-odds attribution breakdowns displaying exact clinical drivers for each prediction.
* **Econometric Cost Optimization Theorem:** Evaluates the economic tradeoff between average 30-day hospital readmission penalties ($16,037) versus preventative care-coordination outreach ($250/patient), deriving the optimal intervention probability cutoff threshold.

### 3. Three-Layer Deduplication Engine
* **Layer 1 (Exact Cryptographic Match):** SHA-256 binary hash check.
* **Layer 2 (Textual Near-Duplicate):** MinHash Jaccard similarity signature with configurable similarity thresholding.
* **Layer 3 (Perceptual Image Hash):** Perceptual hashing (pHash) with Hamming distance verification for visual claim re-submissions.

### 4. Side-by-Side Adjudication & In-App Visual Inspector
* Split-screen workspace rendering the original document binary via `streamlit-pdf-viewer` alongside extracted patient records and calculated risk scores.
* Resilient fallback rendering: automatically formats a clinical document chart if raw binary storage is not populated.
* One-click adjudication decisions: *Approve Claim*, *Flag for Audit*, and *Deny Payout*.

### 5. Retrieval-Augmented Generation (RAG) & Vector Search
* Vector embeddings: Generated via Gemini `text-embedding-004` (768 dimensions).
* Vector similarity search: Powered by PostgreSQL `pgvector` with cosine similarity RPC functions.
* Audit Assistant: Interactive clinical Q&A providing exact patient record citations and grounded source evidence.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend & UI** | Streamlit, Streamlit PDF Viewer, Plotly Graph Objects, CSS3 Custom Design System |
| **Multimodal Vision & LLM** | Google GenAI SDK, Gemini 2.5 Flash, Gemini 3.1 Flash-Lite |
| **Embedding Model** | Google Gemini `text-embedding-004` (768-dimensional vectors) |
| **Database & Vector Store** | Supabase PostgreSQL, `pgvector` Extension |
| **Cloud Object Storage** | Supabase Storage (`clinical-docs` bucket, 1 GB Free Tier) |
| **Machine Learning & Analytics** | Scikit-Learn, NumPy, Pandas, SciPy |
| **Document Processing** | PyPDF, Pillow, ImageHash, Python-Dotenv, Tenacity |

---

## Repository Structure

```
.
|-- app.py                              # Application entrypoint & navigation orchestrator
|-- requirements.txt                    # Project dependencies
|-- README.md                           # Technical documentation
|-- research_and_roi_report.pdf         # Postgraduate research report & econometric analysis
|-- .streamlit/
|   `-- config.toml                     # Streamlit corporate theme tokens & UI layout
|-- views/
|   |-- 0_Upload.py                     # Document ingestion & real-time risk evaluation
|   |-- 1_Search.py                     # Registry search & RAG clinical audit chat
|   |-- 2_Action Centre.py              # Claims queue & side-by-side visual review
|   |-- 3_Settings.py                   # Adjudication rules & custom attribute toggles
|   |-- 5_ML_Predictions.py             # Scenario simulator, ROC curves & economic theorems
|   `-- LOGO.png                        # Platform branding asset
`-- src/
    |-- config/
    |   `-- config.py                   # Environment configuration & credential loader
    |-- extraction/
    |   |-- extraction_service.py       # Dual-engine digital & vision OCR extractor
    |   `-- metadata_service.py         # Structured clinical entity JSON parser
    |-- indexing/
    |   `-- duplicate_detection_service.py # 3-Layer SHA256, MinHash & pHash deduplication
    |-- ml_services/
    |   `-- readmission_predictor.py    # 30-day risk scoring model, XAI & benchmarks
    |-- utils/
    |   |-- get_prompt.py               # Prompt engineering templates for extraction & RAG
    |   |-- gemini_cost_calculator.py   # Token usage & API expenditure tracker
    |   |-- logger.py                   # Operational logging utilities
    |   |-- ocr_scoring.py              # Weighted token OCR confidence algorithm
    |   |-- supabase_client.py          # PostgreSQL CRUD, pgvector RPC & storage API
    |   `-- time_utils.py               # Precision timezone conversion helpers
    `-- validation/
        |-- cross_validation_service.py # Rule-based clinical cross-validation
        |-- document_validation_fields.py # Required schema registry for clinical claims
        |-- file_validator.py           # File extension, size & PDF integrity validation
        `-- validation_engine.py        # Multi-tiered claim validation orchestrator
```

---

## Database Schema (PostgreSQL / Supabase)

The platform requires the following relational schema and `pgvector` configuration:

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Patients Master Table
CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    dob DATE,
    gender VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Clinical Records Table
CREATE TABLE IF NOT EXISTS clinical_records (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) REFERENCES patients(patient_id) ON DELETE CASCADE,
    file_name VARCHAR(255),
    document_number VARCHAR(100),
    entity_name VARCHAR(255),
    admission_date DATE,
    discharge_date DATE,
    length_of_stay INTEGER DEFAULT 0,
    primary_diagnosis TEXT,
    comorbidities TEXT[],
    num_medications INTEGER DEFAULT 0,
    follow_up_scheduled BOOLEAN DEFAULT FALSE,
    lives_alone BOOLEAN DEFAULT FALSE,
    readmission_risk FLOAT,
    adjudication_status VARCHAR(50) DEFAULT 'Pending Review',
    review_reason TEXT,
    extracted_text TEXT,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Deduplication Hashes Table
CREATE TABLE IF NOT EXISTS document_hashes (
    sha256_hash VARCHAR(64) PRIMARY KEY,
    record_id UUID REFERENCES clinical_records(record_id) ON DELETE CASCADE,
    minhash_signature TEXT,
    phash_signature VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cosine Similarity Vector Match Function
CREATE OR REPLACE FUNCTION match_clinical_records (
    query_embedding VECTOR(768),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    record_id UUID,
    patient_id VARCHAR(50),
    file_name VARCHAR(255),
    primary_diagnosis TEXT,
    readmission_risk FLOAT,
    adjudication_status VARCHAR(50),
    extracted_text TEXT,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        clinical_records.record_id,
        clinical_records.patient_id,
        clinical_records.file_name,
        clinical_records.primary_diagnosis,
        clinical_records.readmission_risk,
        clinical_records.adjudication_status,
        clinical_records.extracted_text,
        1 - (clinical_records.embedding <=> query_embedding) AS similarity
    FROM clinical_records
    WHERE 1 - (clinical_records.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

---

## Local Development & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdvector/AI-Powered-Claims-Adjudication-Readmission-Platform.git
cd AI-Powered-Claims-Adjudication-Readmission-Platform
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_anon_public_key
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

This platform is architected for deployment on **Streamlit Community Cloud** with zero local dependencies:

1. Push your repository to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and select this repository.
3. Set the Main file path to `app.py`.
4. In **Advanced Settings > Secrets**, configure:
```toml
GEMINI_API_KEY = "your_google_gemini_api_key"
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your_supabase_anon_public_key"
```
5. Click **Deploy**.

---

## Academic Context & Research

This platform was developed as part of postgraduate research in **Quantitative Economics and Data Science** at **Birla Institute of Technology (BIT) Mesra**.

* **Dissertation Title:** *An AI-Powered Document Intelligence Platform for Hospital Readmission Risk Prediction & Healthcare Claims Adjudication*
* **Core Research Artifact:** See [`research_and_roi_report.pdf`](./research_and_roi_report.pdf) for statistical validation methodologies, calibration benchmarks (AUC-ROC: 0.842), and economic payoff matrices under hospital penalty frameworks.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
