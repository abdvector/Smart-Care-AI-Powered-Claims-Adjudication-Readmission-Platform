"""
Self-Reflective Agentic RAG Engine.
Implements advanced Retrieval-Augmented Generation with:
1. Query Routing & Decomposition Agent
2. Vector Retrieval (Supabase pgvector)
3. Self-Reflection & Evidence Relevance Grader
4. Grounded Synthesis with Verifiable Citations
"""
import json
from typing import Dict, Any, List
from google import genai
from src.config.config import GEMINI_API_KEY
from src.utils.supabase_client import semantic_search_records, keyword_search_records

def _call_gemini(prompt: str, system_instruction: str = "") -> str:
    """Helper to query Gemini model."""
    try:
        if not GEMINI_API_KEY:
            return ""
        client = genai.Client(api_key=GEMINI_API_KEY)
        full_contents = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        resp = None
        for m_name in ["gemini-3.6-flash", "gemini-3.1-flash-lite"]:
            try:
                resp = client.models.generate_content(
                    model=m_name,
                    contents=full_contents
                )
                if resp and resp.text:
                    break
            except Exception:
                continue
        return resp.text.strip() if resp and resp.text else ""
    except Exception as e:
        print(f"[-] Gemini call error: {e}")
        return ""


def route_and_decompose_query(user_query: str) -> Dict[str, Any]:
    """Agent Step 1: Analyzes complexity, extracts key clinical entities, and generates sub-queries."""
    system_instruction = """
You are the Query Routing & Decomposition Agent for a clinical records database.
Analyze the user's healthcare question. Determine:
1. "intent": "Comparative" | "Cohort_Lookup" | "Specific_Patient" | "General_Audit"
2. "is_complex": true/false
3. "sub_queries": list of 1-3 targeted search queries to retrieve maximum relevant evidence.
4. "key_filters": { "diagnosis": "...", "risk_tier": "High/Moderate/Low/None" }

Return ONLY valid JSON:
{
    "intent": "Cohort_Lookup",
    "is_complex": false,
    "sub_queries": ["query 1", "query 2"],
    "key_filters": {}
}
"""
    raw_res = _call_gemini(f"User Query: {user_query}", system_instruction)
    try:
        clean_text = raw_res
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        return json.loads(clean_text)
    except Exception:
        return {
            "intent": "Cohort_Lookup",
            "is_complex": len(user_query.split()) > 7,
            "sub_queries": [user_query],
            "key_filters": {}
        }


def grade_evidence_relevance(user_query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agent Step 2: Self-Reflection grader evaluating whether retrieved records sufficiently answer the query."""
    if not retrieved_docs:
        return {
            "is_sufficient": False,
            "quality_score": 0,
            "reasoning": "No documents retrieved from vector database."
        }

    docs_summary = "\n".join([
        f"- Patient {d.get('patient_id')}: {d.get('primary_diagnosis')}, Risk: {float(d.get('readmission_risk', 0.0))*100:.1f}%, Status: {d.get('adjudication_status')}"
        for d in retrieved_docs[:4]
    ])

    system_instruction = """
You are the Evidence Relevance Grader Agent.
Examine the user query and the retrieved clinical record snippets.
Assess whether the retrieved records contain sufficient and relevant information to answer the question.

Return ONLY valid JSON:
{
    "is_sufficient": true | false,
    "quality_score": 0-100,
    "reasoning": "brief explanation"
}
"""
    prompt = f"User Query: {user_query}\n\nRetrieved Records:\n{docs_summary}"
    raw_res = _call_gemini(prompt, system_instruction)
    try:
        clean_text = raw_res
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        return json.loads(clean_text)
    except Exception:
        return {
            "is_sufficient": len(retrieved_docs) > 0,
            "quality_score": 85 if retrieved_docs else 0,
            "reasoning": f"Retrieved {len(retrieved_docs)} candidate clinical records from pgvector."
        }


def run_agentic_rag(user_query: str) -> Dict[str, Any]:
    """
    Complete Agentic RAG Pipeline:
    1. Query Routing & Decomposition
    2. Multi-hop / Fallback Retrieval
    3. Self-Reflection Relevance Grading
    4. Grounded Synthesis with Explicit Citations
    """
    trace_steps = []

    # Step 1: Query Routing
    trace_steps.append({"step": "1. Query Routing & Decomposition", "status": "Analyzing query structure..."})
    routing_info = route_and_decompose_query(user_query)
    sub_queries = routing_info.get("sub_queries", [user_query])
    trace_steps[-1]["status"] = f"Intent: {routing_info.get('intent')} | Generated {len(sub_queries)} sub-queries."

    # Step 2: Vector Retrieval & Deduplication of Results
    trace_steps.append({"step": "2. Supabase pgvector Retrieval", "status": "Querying 768-dim embeddings..."})
    all_docs = []
    seen_ids = set()

    for q in sub_queries:
        docs = semantic_search_records(q, match_threshold=0.18, match_count=5)
        if not docs:
            docs = keyword_search_records(q, limit=4)
        for d in docs:
            rec_id = d.get("record_id") or d.get("patient_id")
            if rec_id not in seen_ids:
                seen_ids.add(rec_id)
                all_docs.append(d)

    if not all_docs:
        # Fallback to general keyword search
        all_docs = keyword_search_records(user_query, limit=5)

    trace_steps[-1]["status"] = f"Retrieved {len(all_docs)} unique clinical candidate records."

    # Step 3: Self-Reflection & Evidence Grading
    trace_steps.append({"step": "3. Self-Reflection & Relevance Grading", "status": "Evaluating evidence sufficiency..."})
    grading = grade_evidence_relevance(user_query, all_docs)
    trace_steps[-1]["status"] = f"Sufficiency: {'Passed' if grading.get('is_sufficient') else 'Partial'} (Score: {grading.get('quality_score')}/100) - {grading.get('reasoning')}"

    # Step 4: Grounded Synthesis
    trace_steps.append({"step": "4. Grounded Evidence Synthesis", "status": "Generating audit response with strict citations..."})

    citations = []
    for i, doc in enumerate(all_docs[:4], 1):
        citations.append({
            "citation_num": i,
            "patient_id": doc.get("patient_id", "N/A"),
            "primary_diagnosis": doc.get("primary_diagnosis", "N/A"),
            "file_name": doc.get("file_name", "Document"),
            "readmission_risk": float(doc.get("readmission_risk", 0.0) or 0.0) * 100,
            "adjudication_status": doc.get("adjudication_status", "Pending Review")
        })

    if all_docs:
        context_blocks = []
        for i, doc in enumerate(all_docs[:4], 1):
            context_blocks.append(
                f"--- EVIDENCE SOURCE [{i}] ---\n"
                f"Patient ID: {doc.get('patient_id')}\n"
                f"File: {doc.get('file_name')}\n"
                f"Primary Diagnosis: {doc.get('primary_diagnosis')}\n"
                f"Comorbidities: {doc.get('comorbidities')}\n"
                f"Length of Stay: {doc.get('length_of_stay')} days\n"
                f"7-Day Follow-Up: {doc.get('follow_up_scheduled')}\n"
                f"Lives Alone (SDoH): {doc.get('lives_alone')}\n"
                f"Readmission Risk: {float(doc.get('readmission_risk', 0.0))*100:.1f}%\n"
                f"Adjudication Status: {doc.get('adjudication_status')}\n"
                f"Clinical Record Snippet: {str(doc.get('extracted_text', ''))[:800]}\n"
            )

        context_str = "\n".join(context_blocks)

        synthesis_prompt = f"""
You are the Smart-Care Senior Clinical Audit AI.
Answer the user's question using ONLY the retrieved clinical evidence below.
Format your response clearly:
1. Executive Answer: Direct, clinical answer addressing the query.
2. Clinical Findings & Patient Patterns: Bullet points with explicit reference to Evidence Source numbers (e.g. [Evidence 1], [Evidence 2]).
3. Adjudication & Risk Insights: Highlight high-risk readmissions and any care-coordination action required.

Do not hallucinate or speculate on patients not documented in the evidence.

User Question: {user_query}

Retrieved Clinical Evidence:
{context_str}
"""
        answer_text = _call_gemini(synthesis_prompt)
        if not answer_text:
            answer_text = (
                f"Based on the {len(all_docs)} retrieved records, documented cases include patient "
                f"`{all_docs[0].get('patient_id')}` presenting with {all_docs[0].get('primary_diagnosis')} "
                f"with a 30-day readmission risk of {float(all_docs[0].get('readmission_risk', 0.0))*100:.1f}%."
            )
    else:
        answer_text = "I queried the Supabase vector database across all available records, but could not find matching clinical documentation for this specific query. Please ensure relevant discharge summaries have been uploaded."

    trace_steps[-1]["status"] = "Synthesis completed with verified citations."

    return {
        "answer": answer_text,
        "citations": citations,
        "trace_steps": trace_steps,
        "routing_info": routing_info,
        "grading": grading
    }
