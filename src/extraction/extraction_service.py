"""
Free Multi-Format Text Extraction Service
Extracts text from digital PDFs via pypdf and scanned images via Gemini Vision Flash.
"""
import io
import os
from pypdf import PdfReader
from google import genai
from google.genai import types
from src.config.config import GEMINI_API_KEY

class PageObject:
    def __init__(self, page_number: int, content: str = ""):
        self.page_number = page_number
        self.content = content
        self.words = []

class ExtractionResult:
    def __init__(self, content: str = "", pages: list = None, confidence: float = 95.0):
        self.content = content or ""
        self.pages = pages or [PageObject(1, content)]
        self.confidence = confidence

def get_gemini_client():
    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if key:
        return genai.Client(api_key=key)
    return None

def extract_text(file_input, extension: str = None) -> ExtractionResult:
    """
    Extracts text using free digital parsing (pypdf) or Gemini Vision for scanned images.
    Accepts file path (str) or raw bytes.
    """
    print("\n[EXTRACTION] Processing document for text extraction...")
    
    # Read bytes if string path
    if isinstance(file_input, str):
        with open(file_input, "rb") as f:
            file_bytes = f.read()
        if not extension:
            extension = os.path.splitext(file_input)[1].lower()
    else:
        file_bytes = file_input

    ext = (extension or ".pdf").lower()
    extracted_text = ""
    pages = []

    # 1. Digital PDF extraction via pypdf
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages.append(PageObject(i + 1, page_text))
                extracted_text += page_text + "\n"
            
            extracted_text = extracted_text.strip()
            if len(extracted_text) > 30:
                print(f"[EXTRACTION] Successfully extracted {len(pages)} pages via pypdf.")
                return ExtractionResult(content=extracted_text, pages=pages, confidence=98.5)
            else:
                print("[EXTRACTION] PDF contains little/no digital text (likely scanned). Falling back to Gemini Vision...")
        except Exception as e:
            print(f"[-] pypdf extraction error: {e}")

    # 2. DOCX extraction
    if ext in [".docx", ".doc"]:
        try:
            import mammoth
            result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
            extracted_text = result.value.strip()
            return ExtractionResult(content=extracted_text, pages=[PageObject(1, extracted_text)], confidence=99.0)
        except Exception as e:
            print(f"[-] DOCX extraction error: {e}")

    # 3. Scanned PDF or Image extraction via Gemini Multimodal (Free Tier)
    try:
        client = get_gemini_client()
        if client:
            mime_type = "application/pdf" if ext == ".pdf" else "image/png"
            if ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
                
            prompt = "Extract all text and tabular information from this clinical document accurately and verbatim."
            
            response = None
            for m_name in ["gemini-3.6-flash", "gemini-3.1-flash-lite"]:
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=[
                            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                            prompt
                        ]
                    )
                    if response and response.text:
                        break
                except Exception:
                    continue
            if response and response.text:
                extracted_text = response.text.strip()
                print("[EXTRACTION] Gemini Vision successfully extracted scanned document text.")
                return ExtractionResult(content=extracted_text, pages=[PageObject(1, extracted_text)], confidence=94.0)
    except Exception as e:
        print(f"[-] Gemini Vision OCR failed: {e}")

    # Fallback if empty
    return ExtractionResult(content=extracted_text or "No text could be extracted.", pages=[PageObject(1, extracted_text)], confidence=50.0)
