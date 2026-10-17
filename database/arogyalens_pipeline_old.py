"""
ArogyaLens — Combined Classification Pipeline (v2)
Tier 1 (report-provided range) + Tier 2 (local PostgreSQL DB) + Tier 3 (Search-RAG)
with Grok-based AI cross-verification, sex/age-aware Tier 3, unit normalization,
and patient result storage.

Run: python arogyalens_pipeline.py
"""

import re
import requests
import time
import json
import xml.etree.ElementTree as ET
import psycopg2
import google.generativeai as genai
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = dict(dbname=os.getenv("DB_NAME", "arogyalens"), user=os.getenv("DB_USER"),
                 password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST", "localhost"))

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VERIFIER_API_KEY = os.getenv("VERIFIER_API_KEY") or "your_verifier_key_here"        # <-- provider ki key yahan
VERIFIER_BASE_URL = "https://openrouter.ai/api/v1"  # OpenRouter. Mistral: https://api.mistral.ai/v1
VERIFIER_MODEL = "openai/gpt-oss-120b:free" # provider ke Models page se current naam lo

genai.configure(api_key=GEMINI_API_KEY)
# LLM verifier OPTIONAL hai: key bhari ho tabhi chalega, warna sirf rule-based
# number-in-source check use hoga (koi API/billing nahi chahiye).
LLM_VERIFIER_ENABLED = VERIFIER_API_KEY not in ("", "your_verifier_key_here")
verifier_client = (OpenAI(api_key=VERIFIER_API_KEY, base_url=VERIFIER_BASE_URL)
                   if LLM_VERIFIER_ENABLED else None)

GEMINI_MODEL = "gemini-3.8-flash"   # confirmed working model


def _clean_json(text):
    """LLM ke ```json ... ``` fences hata deta hai."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    return t


# ============================================================
# HELPERS — age band + UNIT NORMALIZATION
# ============================================================
def age_band(age):
    if age is None:
        return "adult"
    if age < 18:
        return "child"
    if age >= 65:
        return "senior"
    return "adult"


# --- UNIT_START ---
def _norm_unit(u):
    if not u:
        return ""
    u = str(u).strip().lower()
    u = u.replace("µ", "u").replace("μ", "u").replace(" ", "")
    u = u.replace("×", "x").replace("**", "^")
    return u

# dimension -> {unit: factor to that dimension's base unit}
UNIT_TABLE = {
    # base = cells per microlitre (/uL == /cumm == /mm3)
    "count_per_ul": {
        "/ul": 1, "cells/ul": 1, "/cumm": 1, "cells/cumm": 1, "cumm": 1,
        "/mm3": 1, "cells/mm3": 1, "/cmm": 1,
        "10^3/ul": 1e3, "x10^3/ul": 1e3, "10*3/ul": 1e3, "x10*3/ul": 1e3,
        "10^3/cumm": 1e3, "x10^3/cumm": 1e3, "10^3/mm3": 1e3, "10^3/cmm": 1e3,
        "thousand/cumm": 1e3, "thousands/cumm": 1e3, "k/ul": 1e3,
        "10^9/l": 1e3, "x10^9/l": 1e3,
        "lakh/cumm": 1e5, "lakhs/cumm": 1e5, "lac/cumm": 1e5, "lacs/cumm": 1e5,
        "lakh/ul": 1e5, "lakh/cmm": 1e5,
        "million/cumm": 1e6, "millions/cumm": 1e6, "mill/cumm": 1e6, "mil/cumm": 1e6,
        "10^6/ul": 1e6, "x10^6/ul": 1e6, "10*6/ul": 1e6, "10^6/cumm": 1e6,
        "x10^6/cumm": 1e6, "m/ul": 1e6,
        "10^12/l": 1e6, "x10^12/l": 1e6,
    },
    # base = g/dL
    "mass_per_dl": {
        "g/dl": 1, "gm/dl": 1, "gm%": 1, "g%": 1, "mg/dl": 1e-3,
        "g/l": 0.1, "mg/l": 1e-4, "ug/dl": 1e-6, "mcg/dl": 1e-6,
    },
    # base = ng/mL
    "conc_ng_ml": {
        "ng/ml": 1, "ug/l": 1, "mcg/l": 1, "pg/ml": 1e-3, "ng/l": 1e-3,
        "ug/ml": 1e3, "mcg/ml": 1e3, "ng/dl": 1e-2,
    },
    # base = uIU/mL
    "miu": {"uiu/ml": 1, "miu/l": 1, "miu/ml": 1e3, "uiu/l": 1e-3},
    "enzyme": {"u/l": 1, "iu/l": 1},
    "percent": {"%": 1},
    "fl": {"fl": 1},
    "pg": {"pg": 1},
    "mmol_l": {"mmol/l": 1},
}
_UNIT_LOOKUP = {u: (dim, f) for dim, d in UNIT_TABLE.items() for u, f in d.items()}


def convert_value(value, from_unit, to_unit):
    """value ko from_unit se to_unit mein badalta hai.
    Returns (value, status): status = same | converted | unchecked | mismatch"""
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if not f or not t:
        return value, "unchecked"          # unit pata nahi, check nahi ho sakta
    if f == t:
        return value, "same"
    if f in _UNIT_LOOKUP and t in _UNIT_LOOKUP:
        fd, ff = _UNIT_LOOKUP[f]
        td, tf = _UNIT_LOOKUP[t]
        if fd == td:
            return round(float(value) * ff / tf, 6), "converted"
    return None, "mismatch"                # units compatible nahi / pata nahi
# --- UNIT_END ---


# ============================================================
# TIER 1 — Report-provided reference range
# ============================================================
def validate_report_range(low, high):
    """Report ke apne printed range ko validate karta hai.
    One-sided range bhi valid hai: '<200' -> (None, 200), '>40' -> (40, None)."""
    if low is None and high is None:
        return None
    try:
        low = float(low) if low is not None else None
        high = float(high) if high is not None else None
    except (TypeError, ValueError):
        return None
    if low is not None and high is not None and low >= high:
        return None
    return {'low': low, 'high': high, 'source': 'report_ocr'}


# ============================================================
# TIER 2 — Local curated PostgreSQL database
# ============================================================
def get_local_db_range(cursor, test_name, sex, age):
    cursor.execute("""
        SELECT lt.canonical_name, rr.sex, rr.low_value, rr.high_value, rr.unit
        FROM reference_ranges rr
        JOIN lab_tests lt ON lt.test_id = rr.test_id
        LEFT JOIN lab_test_aliases lta ON lta.test_id = lt.test_id
        WHERE (lt.canonical_name = %s OR lta.alias_text = %s)
          AND rr.sex IN (%s, 'any')
          AND %s BETWEEN rr.age_min AND rr.age_max
        LIMIT 1
    """, (test_name, test_name, sex, age))
    row = cursor.fetchone()
    if row:
        return {'low': row[2], 'high': row[3], 'unit': row[4], 'source': 'local_db'}
    return None


# ============================================================
# TIER 3 — Search-RAG — PRIMARY SOURCES (real APIs)
# ============================================================
def search_pubmed(query, max_results=3):
    params = {"db": "pubmed", "term": f"{query} normal reference range",
              "retmax": max_results, "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=10)
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    time.sleep(0.4)
    fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
    if NCBI_API_KEY:
        fetch_params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=fetch_params, timeout=10)
    return [{"text": r.text, "url": f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/", "source": "PubMed"}]


def search_pmc(query, max_results=2):
    params = {"db": "pmc", "term": f"{query} reference range",
              "retmax": max_results, "retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=10)
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    time.sleep(0.4)
    fetch_params = {"db": "pmc", "id": ",".join(ids), "rettype": "full", "retmode": "text"}
    if NCBI_API_KEY:
        fetch_params["api_key"] = NCBI_API_KEY
    r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=fetch_params, timeout=10)
    return [{"text": r.text[:5000], "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{ids[0]}/", "source": "NIH (PMC)"}]


def search_medlineplus(query):
    params = {"db": "healthTopics", "term": query}
    resp = requests.get("https://wsearch.nlm.nih.gov/ws/query", params=params, timeout=10)
    root = ET.fromstring(resp.content)
    docs = root.findall(".//document")
    if not docs:
        return []
    doc = docs[0]
    url = doc.get("url", "")
    parts = [c.text or "" for c in doc.findall("content") if c.get("name") == "FullSummary"]
    text = " ".join(parts)[:3000]
    return [{"text": text, "url": url, "source": "MedlinePlus"}] if text else []


def search_fda(medicine_name):
    resp = requests.get("https://api.fda.gov/drug/label.json",
                        params={"search": f'openfda.generic_name:"{medicine_name}"', "limit": 1},
                        timeout=10)
    data = resp.json()
    if "results" not in data:
        return []
    result = data["results"][0]
    text_parts = result.get("indications_and_usage", []) + result.get("warnings", [])
    text = " ".join(text_parts)[:3000]
    return [{"text": text, "url": f"FDA drug label: {medicine_name}", "source": "FDA"}] if text else []


# ---------- SUPPORTING SOURCES (Serper site-restricted search) ----------
def search_trusted_site(query, site_domain, source_label, max_results=2):
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": f"{query} site:{site_domain}", "num": max_results},
        timeout=10
    )
    results = resp.json().get("organic", [])
    sources = []
    for r in results:
        try:
            page = requests.get(r["link"], timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            sources.append({"text": page.text[:3000], "url": r["link"], "source": source_label})
        except Exception:
            continue
    return sources

def search_who(query):
    return search_trusted_site(f"{query} reference range", "who.int", "WHO")

def search_mayo(query):
    return search_trusted_site(f"{query} reference range", "mayocliniclabs.com", "Mayo Clinic Labs")

def search_arup(query):
    return search_trusted_site(f"{query} reference interval", "arupconsult.com", "ARUP Labs")


def gather_trusted_sources(query_name, query_type="lab_test", extra=""):
    """query_type: 'lab_test' or 'medicine'.
    extra: demographic text (jaise 'male adult') jo lab_test searches mein jodta hai."""
    results = []
    q = f"{query_name} {extra}".strip()
    if query_type == "lab_test":
        primary = [("PubMed", lambda: search_pubmed(q)),
                   ("NIH/PMC", lambda: search_pmc(q)),
                   ("MedlinePlus", lambda: search_medlineplus(query_name))]
        supporting = [("Mayo Clinic Labs", lambda: search_mayo(q)),
                      ("ARUP Labs", lambda: search_arup(q)),
                      ("WHO", lambda: search_who(q))]
    else:
        primary = [("FDA", lambda: search_fda(query_name)),
                   ("MedlinePlus", lambda: search_medlineplus(query_name))]
        supporting = [("PubMed", lambda: search_pubmed(query_name)),
                      ("WHO", lambda: search_who(query_name))]

    for label, fn in primary:
        try:
            r = fn()
            print(f"[primary] {label}: {len(r)} results")
            results += r
        except Exception as e:
            print(f"[primary] {label} fail: {e}")

    for label, fn in supporting:
        try:
            r = fn()
            print(f"[supporting] {label}: {len(r)} results")
            results += r
        except Exception as e:
            print(f"[supporting] {label} fail: {e}")

    return results


# ---------- Gemini extraction ----------
def extract_range_from_sources(test_name, sources, sex="any", band="adult"):
    combined_text = "\n\n---\n\n".join([s["text"][:2000] for s in sources])
    who = f"sex: {sex}, age group: {band}"
    prompt = f"""
Extract a clinical reference range from trusted medical text.
Test: {test_name}
Patient: {who}
Source text:
{combined_text}

Rules:
- If the text gives separate ranges by sex or age, pick the one matching the patient.
- If the text gives only one general range (no sex/age split), return it and set "applies_to" to "general".
- If the text clearly gives a range only for a DIFFERENT group than the patient, return all nulls.
- Never guess a number that is not in the text.

Return ONLY valid JSON, no markdown:
{{"low": <number or null>, "high": <number or null>, "unit": "<string or null>",
  "applies_to": "<specific or general>", "confidence": "<high/medium/low>"}}
If no clear numeric range exists, return all nulls.
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    try:
        return json.loads(_clean_json(response.text))
    except json.JSONDecodeError:
        return {"low": None, "high": None, "unit": None, "confidence": "low"}


def extract_medicine_info(medicine_name, sources):
    combined_text = "\n\n---\n\n".join([s["text"][:2000] for s in sources])
    prompt = f"""
Extract medicine information from trusted medical sources.
Medicine: {medicine_name}
Source text:
{combined_text}

Return ONLY valid JSON, no markdown:
{{
  "common_use": "<plain-language summary, or null>",
  "drug_class": "<drug class, or null>",
  "is_antibiotic": <true/false>,
  "who_aware_class": "<Access/Watch/Reserve, or null>",
  "key_warning": "<one important warning, or null>",
  "confidence": "<high/medium/low>"
}}
"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    try:
        return json.loads(_clean_json(response.text))
    except json.JSONDecodeError:
        return {"common_use": None, "confidence": "low"}


# ---------- Grok cross-check (hallucination verification) ----------
def verify_extracted_range(test_name, extracted, sources, sex="any", band="adult"):
    """Doosra, independent model (Grok) check karta hai ki Gemini ka extraction
    sach mein source text mein tha, aur patient (sex/age) par lagu hota hai."""
    if extracted.get("low") is None:
        return {**extracted, "verified": False, "verification_note": "No range was extracted"}
    if not LLM_VERIFIER_ENABLED:
        return {**extracted, "verified": None, "verification_note": "LLM verifier disabled (no key)"}

    combined_text = "\n\n---\n\n".join([s["text"][:2000] for s in sources])
    prompt = f"""You are a strict fact-checker. Below is medical source text and a claimed reference range.
Test: {test_name}
Patient: sex={sex}, age group={band}
Claimed range: low={extracted.get('low')}, high={extracted.get('high')}, unit={extracted.get('unit')}

Source text:
{combined_text}

Check two things:
1. The numbers must genuinely appear in, or be directly derivable from, the source text.
2. The range must apply to this patient: either the text gives it for this sex/age group,
   or the text gives a general range with no sex/age split. If the text gives this range
   only for a different group, answer false.

Return ONLY valid JSON, no markdown, no explanation outside JSON:
{{"verified": true or false, "reason": "one short sentence"}}"""

    try:
        response = verifier_client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = json.loads(_clean_json(response.choices[0].message.content))
    except Exception as e:
        print(f"Grok verification error: {e}")
        result = {"verified": None, "reason": "Verification call failed"}   # None = verifier chala hi nahi (reject nahi)

    return {**extracted, "verified": result.get("verified"), "verification_note": result.get("reason")}


def verify_medicine_info(medicine_name, extracted, sources):
    if not extracted.get("common_use"):
        return {**extracted, "verified": False, "verification_note": "No info was extracted"}
    if not LLM_VERIFIER_ENABLED:
        return {**extracted, "verified": None, "verification_note": "LLM verifier disabled (no key)"}

    combined_text = "\n\n---\n\n".join([s["text"][:2000] for s in sources])
    prompt = f"""You are a strict fact-checker. Below is medical source text and claimed medicine information.
Medicine: {medicine_name}
Claimed info: {json.dumps(extracted)}

Source text:
{combined_text}

Does the source text support this claimed information? Return ONLY valid JSON:
{{"verified": true or false, "reason": "one short sentence"}}"""

    try:
        response = verifier_client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = json.loads(_clean_json(response.choices[0].message.content))
    except Exception as e:
        print(f"Grok verification error: {e}")
        result = {"verified": None, "reason": "Verification call failed"}   # None = verifier chala hi nahi (reject nahi)

    return {**extracted, "verified": result.get("verified"), "verification_note": result.get("reason")}


# ---------- Rule-based verification (koi API nahi) ----------
def _numbers_in_text(text):
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags hatao (width=100 jaise numbers se bachne ke liye)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)         # 150,000 -> 150000
    return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def verify_numbers_in_sources(extracted, sources):
    """Gemini ke nikale low/high numbers source text mein literally maujood hone chahiye.
    Returns (ok, note). Ye hallucinated numbers pakadta hai; sex/age ya doosre test ka
    number nahi pakad sakta (wo LLM verifier ka kaam hai)."""
    found = set()
    for s in sources:
        found |= _numbers_in_text(s["text"][:2000])         # jitna Gemini ne dekha utna hi
    missing = []
    for k in ("low", "high"):
        v = extracted.get(k)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False, f"{k} number nahi hai"
        if not any(abs(v - f) < 1e-9 for f in found):
            missing.append(v)
    if missing:
        return False, f"numbers source text mein nahi mile: {missing}"
    return True, "numbers source text mein maujood"


# ---------- Tier 3 master functions (cache-first -> live search -> verify) ----------
def rag_cache_key(test_name, sex, age):
    """Cache key mein sex + age band shamil, taaki male/female/child ke ranges mix na hon."""
    return f"{test_name}|{sex or 'any'}|{age_band(age)}"


def get_rag_reference_range(cursor, test_name, sex="any", age=None):
    key = rag_cache_key(test_name, sex, age)
    band = age_band(age)

    cursor.execute(
        "SELECT retrieved_low, retrieved_high, retrieved_unit, confidence FROM rag_lookup_cache WHERE query_text = %s",
        (key,)
    )
    row = cursor.fetchone()
    if row and row[3] != 'rejected':
        cursor.execute("UPDATE rag_lookup_cache SET hit_count = hit_count + 1 WHERE query_text = %s", (key,))
        return {'low': row[0], 'high': row[1], 'unit': row[2], 'source': 'rag_cached'}

    demo = f"{'' if sex in (None, 'any') else sex} {band}".strip()
    sources = gather_trusted_sources(test_name, query_type="lab_test", extra=demo)
    if not sources:
        return {'low': None, 'high': None, 'source': 'unavailable'}

    extracted = extract_range_from_sources(test_name, sources, sex or "any", band)
    if extracted.get("low") is None:
        return {'low': None, 'high': None, 'source': 'unavailable'}

    # Layer 1: rule-based (hamesha chalta hai, koi API nahi)
    ok, note = verify_numbers_in_sources(extracted, sources)
    if not ok:
        print(f"⚠️ Number check failed for {test_name} ({demo}): {note}")
        return {'low': None, 'high': None, 'source': 'unavailable'}

    # Layer 2: LLM verifier (optional) - sex/age aur context check
    verified_result = verify_extracted_range(test_name, extracted, sources, sex or "any", band)
    llm_verdict = verified_result.get("verified")
    if llm_verdict is False:
        print(f"⚠️ LLM verifier ne reject kiya: {test_name} ({demo}): {verified_result.get('verification_note')}")
        return {'low': None, 'high': None, 'source': 'unavailable'}
    if llm_verdict is None:
        print(f"ℹ️ LLM verifier nahi chala ({verified_result.get('verification_note')}); sirf number-in-source check par bharosa")
    ai_flag = llm_verdict is True

    urls = [s["url"] for s in sources if s.get("url")]
    cursor.execute("""
        INSERT INTO rag_lookup_cache (query_text, retrieved_low, retrieved_high, retrieved_unit, source_urls, confidence, ai_verified)
        VALUES (%s, %s, %s, %s, %s, 'unverified', %s)
        ON CONFLICT (query_text) DO NOTHING
    """, (key, extracted["low"], extracted["high"], extracted.get("unit"), urls, ai_flag))

    return {'low': extracted["low"], 'high': extracted["high"], 'unit': extracted.get("unit"),
            'source': 'rag_live', 'applies_to': extracted.get("applies_to")}


def get_rag_medicine_info(cursor, medicine_name):
    cursor.execute(
        "SELECT common_use, drug_class, is_antibiotic, who_aware_class, key_warning, confidence FROM rag_medicine_cache WHERE query_text = %s",
        (medicine_name,)
    )
    row = cursor.fetchone()
    if row and row[5] != 'rejected':
        cursor.execute("UPDATE rag_medicine_cache SET hit_count = hit_count + 1 WHERE query_text = %s", (medicine_name,))
        return {'common_use': row[0], 'drug_class': row[1], 'is_antibiotic': row[2],
                'who_aware_class': row[3], 'key_warning': row[4], 'source': 'rag_cached'}

    sources = gather_trusted_sources(medicine_name, query_type="medicine")
    if not sources:
        return {'common_use': None, 'source': 'unavailable'}

    extracted = extract_medicine_info(medicine_name, sources)
    if not extracted.get("common_use"):
        return {'common_use': None, 'source': 'unavailable'}

    verified_result = verify_medicine_info(medicine_name, extracted, sources)
    if not verified_result.get("verified"):
        print(f"⚠️ Grok verification failed for {medicine_name}: {verified_result.get('verification_note')}")
        return {'common_use': None, 'source': 'unavailable'}

    urls = [s["url"] for s in sources if s.get("url")]
    cursor.execute("""
        INSERT INTO rag_medicine_cache
            (query_text, common_use, drug_class, is_antibiotic, who_aware_class, key_warning, source_urls, confidence, ai_verified)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'unverified', %s)
        ON CONFLICT (query_text) DO NOTHING
    """, (medicine_name, extracted.get("common_use"), extracted.get("drug_class"),
          extracted.get("is_antibiotic"), extracted.get("who_aware_class"),
          extracted.get("key_warning"), urls, True))

    return {**extracted, 'source': 'rag_live'}


# ============================================================
# MASTER — Tier 1 + Tier 2 + Tier 3, ek saath
# ============================================================
def classify_value(cursor, test_name, value, sex, age,
                   report_low=None, report_high=None, unit=None):
    """unit = report mein value ki unit (OCR se). Range ki unit se alag ho toh
    convert hota hai; convert na ho sake toh 'unit_mismatch' (andaza nahi lagata)."""
    # Tier 1
    range_result = validate_report_range(report_low, report_high)
    if range_result is not None:
        range_result['unit'] = unit          # report ka range report ki apni unit mein hota hai

    # Tier 2
    if range_result is None:
        range_result = get_local_db_range(cursor, test_name, sex, age)

    # Tier 3
    if range_result is None:
        range_result = get_rag_reference_range(cursor, test_name, sex, age)

    if range_result.get('low') is None and range_result.get('high') is None:
        return {'test_name': test_name, 'value': value, 'classification': 'unknown',
                'source': range_result.get('source', 'unavailable'), 'unit': None,
                'low': None, 'high': None, 'reported_value': value, 'reported_unit': unit}

    # ---- UNIT CHECK ----
    check_value, unit_status = convert_value(value, unit, range_result.get('unit'))
    if unit_status == "mismatch":
        return {'test_name': test_name, 'value': value, 'classification': 'unit_mismatch',
                'source': range_result.get('source'), 'unit': range_result.get('unit'),
                'low': range_result.get('low'), 'high': range_result.get('high'),
                'reported_value': value, 'reported_unit': unit}

    low = float(range_result['low']) if range_result.get('low') is not None else None
    high = float(range_result['high']) if range_result.get('high') is not None else None

    # one-sided ranges: '<200' (low=None) aur '>40' (high=None) dono handle
    if low is not None and check_value < low:
        classification = 'low'
    elif high is not None and check_value > high:
        classification = 'high'
    else:
        classification = 'normal'

    return {**range_result, 'test_name': test_name, 'value': check_value,
            'classification': classification, 'reported_value': value,
            'reported_unit': unit, 'unit_status': unit_status}


# ============================================================
# PATIENT REPORT DATA — Postgres mein save karna
# ============================================================
def store_result(cursor, report_id, classified_result):
    cursor.execute("""
        INSERT INTO classified_results
            (report_id, test_name, value, unit, reference_low, reference_high,
             source, classification, reported_value, reported_unit)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        report_id,
        classified_result['test_name'],
        classified_result['value'],
        classified_result.get('unit'),
        classified_result.get('low'),
        classified_result.get('high'),
        classified_result.get('source'),
        classified_result.get('classification'),
        classified_result.get('reported_value'),
        classified_result.get('reported_unit'),
    ))


# ============================================================
# TEST — scenarios, har tier + unit + sex/age ki verification ke saath
# Run: python arogyalens_pipeline.py
# ============================================================
def tier2_info(cursor, test_name, sex, age):
    """Tier 2 range ka source aur kisne verify kiya (DB se)."""
    cursor.execute("""
        SELECT rr.low_value, rr.high_value, rr.source, rr.verified_by
        FROM reference_ranges rr
        JOIN lab_tests lt ON lt.test_id = rr.test_id
        LEFT JOIN lab_test_aliases lta ON lta.test_id = lt.test_id
        WHERE (lt.canonical_name = %s OR lta.alias_text = %s)
          AND rr.sex IN (%s, 'any') AND %s BETWEEN rr.age_min AND rr.age_max
        LIMIT 1
    """, (test_name, test_name, sex, age))
    return cursor.fetchone()


def check(label, ok):
    print(f"   {'PASS' if ok else 'FAIL'} | {label}")
    return ok


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol * max(1, abs(b))


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    report_id = 1001
    passed = []

    cursor.execute("DELETE FROM classified_results WHERE report_id = %s", (report_id,))   # clean run

    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 1 - Report mein range hai  ->  TIER 1")
    print("=" * 60)
    r = classify_value(cursor, "Hemoglobin", 11.2, "male", 23,
                       report_low=13.0, report_high=17.0, unit="g/dL")
    store_result(cursor, report_id, r)
    print("   Result:", r)
    passed.append(check("source = report_ocr", r["source"] == "report_ocr"))
    passed.append(check("classification = low (11.2 < 13.0)", r["classification"] == "low"))
    passed.append(check("report ka range use hua (13.0-17.0)", r["low"] == 13.0 and r["high"] == 17.0))
    db = tier2_info(cursor, "Hemoglobin", "male", 23)
    if db:
        print(f"   Cross-check: report range 13.0-17.0 vs curated DB {db[0]}-{db[1]} ({db[2]})")
        overlap = max(13.0, float(db[0])) < min(17.0, float(db[1]))
        passed.append(check("report range curated DB se overlap karta hai", overlap))

    print("\n" + "-" * 60)
    print("SCENARIO 1b - One-sided range ('<200', OCR ne low=None, high=200 diya)  ->  TIER 1")
    print("-" * 60)
    r = classify_value(cursor, "LDL Cholesterol", 250, "male", 23,
                       report_low=None, report_high=200, unit="mg/dL")
    store_result(cursor, report_id, r)
    print("   Result:", r)
    passed.append(check("one-sided range Tier 1 ne accept kiya", r["source"] == "report_ocr"))
    passed.append(check("250 > 200 -> high", r["classification"] == "high"))
    r = classify_value(cursor, "LDL Cholesterol", 150, "male", 23,
                       report_low=None, report_high=200, unit="mg/dL")
    passed.append(check("150 < 200 -> normal", r["classification"] == "normal"))

    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 2a - Report mein range NAHI  ->  Tier 1 fail, TIER 2")
    print("=" * 60)
    r = classify_value(cursor, "Hemoglobin", 11.2, "male", 23)
    store_result(cursor, report_id, r)
    print("   Result:", r)
    passed.append(check("source = local_db", r["source"] == "local_db"))
    passed.append(check("classification = low", r["classification"] == "low"))
    db = tier2_info(cursor, "Hemoglobin", "male", 23)
    print(f"   Tier 2 verification: source={db[2]}, verified_by={db[3]}")
    passed.append(check("curated row ka source + verified_by DB mein maujood", bool(db[2]) and bool(db[3])))

    print("\n" + "-" * 60)
    print("SCENARIO 2b - Report ka range GALAT (low >= high)  ->  Tier 1 reject, TIER 2")
    print("-" * 60)
    r = classify_value(cursor, "Hemoglobin", 15.0, "male", 23, report_low=18, report_high=12)
    store_result(cursor, report_id, r)
    print("   Result:", r)
    passed.append(check("bekar range reject hua, source = local_db", r["source"] == "local_db"))

    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 3 - Range NAHI + local DB mein bhi NAHI  ->  TIER 3 (+ Grok), male adult")
    print("=" * 60)
    T3 = "Vitamin D"            # agar yeh local DB mein hai toh koi aur naam daalo
    passed.append(check(f"'{T3}' local DB mein nahi hai (Tier 2 fail confirm)",
                        get_local_db_range(cursor, T3, "male", 30) is None))
    cursor.execute("DELETE FROM rag_lookup_cache WHERE query_text LIKE %s", (T3 + "|%",))  # fresh live run
    conn.commit()

    r = classify_value(cursor, T3, 14, "male", 30, unit="ng/mL")
    store_result(cursor, report_id, r)
    conn.commit()
    print("   Result:", r)
    passed.append(check("source = rag_live", r["source"] == "rag_live"))
    passed.append(check("classification mila (unknown/unit_mismatch nahi)",
                        r["classification"] in ("low", "normal", "high")))

    key_m = rag_cache_key(T3, "male", 30)
    cursor.execute("""SELECT retrieved_low, retrieved_high, confidence, ai_verified, hit_count
                      FROM rag_lookup_cache WHERE query_text = %s""", (key_m,))
    row = cursor.fetchone()
    print(f"   Cache row [{key_m}]:", row)
    passed.append(check(f"verification flag sahi (LLM verifier {'ON' if LLM_VERIFIER_ENABLED else 'OFF'} "
                        f"-> ai_verified = {LLM_VERIFIER_ENABLED})",
                        bool(row) and row[3] is LLM_VERIFIER_ENABLED))
    passed.append(check("confidence = unverified (human review baaki)", bool(row) and row[2] == "unverified"))

    print("\n" + "-" * 60)
    print("SCENARIO 3b - Wahi test, wahi patient type dobara  ->  CACHE")
    print("-" * 60)
    r = classify_value(cursor, T3, 14, "male", 30, unit="ng/mL")
    print("   Result:", r)
    passed.append(check("source = rag_cached", r["source"] == "rag_cached"))

    print("\n" + "-" * 60)
    print("SCENARIO 3c - Wahi test, FEMALE  ->  alag cache key, naya live search")
    print("-" * 60)
    r = classify_value(cursor, T3, 14, "female", 30, unit="ng/mL")
    store_result(cursor, report_id, r)
    conn.commit()
    print("   Result:", r)
    passed.append(check("female ke liye naya live search hua (rag_live)", r["source"] == "rag_live"))
    cursor.execute("SELECT query_text FROM rag_lookup_cache WHERE query_text LIKE %s ORDER BY 1", (T3 + "|%",))
    keys = [k[0] for k in cursor.fetchall()]
    print("   Cache keys:", keys)
    passed.append(check("male aur female ki alag cache rows", len(keys) == 2))

    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 4 - UNIT normalization")
    print("=" * 60)
    v, s = convert_value(8500, "cells/µL", "10^3/µL")
    passed.append(check("8500 cells/uL -> 8.5 x10^3/uL", close(v, 8.5) and s == "converted"))
    v, s = convert_value(2.38, "lakh/cumm", "10^3/uL")
    passed.append(check("2.38 lakh/cumm -> 238 x10^3/uL", close(v, 238.0) and s == "converted"))
    v, s = convert_value(1200, "mg/dL", "g/dL")
    passed.append(check("1200 mg/dL -> 1.2 g/dL", close(v, 1.2) and s == "converted"))
    v, s = convert_value(5, "mmol/L", "mg/dL")
    passed.append(check("mmol/L vs mg/dL -> mismatch (andaza nahi)", v is None and s == "mismatch"))

    r = classify_value(cursor, "Hemoglobin", 11.2, "male", 23, unit="mmol/L")
    store_result(cursor, report_id, r)
    print("   Result:", r)
    passed.append(check("Hemoglobin mmol/L -> 'unit_mismatch' (galat classify nahi)",
                        r["classification"] == "unit_mismatch"))

    pdb = get_local_db_range(cursor, "Platelet Count", "any", 23)
    if pdb:
        conv, st = convert_value(238000, "/cumm", pdb["unit"])
        print(f"   Platelet DB unit = {pdb['unit']!r}")
        if conv is None:
            print("   SKIP | Platelet DB unit conversion table mein nahi hai "
                  "(UNIT_TABLE mein add karo)")
        else:
            a = classify_value(cursor, "Platelet Count", 238000, "any", 23, unit="/cumm")
            b = classify_value(cursor, "Platelet Count", conv, "any", 23)
            store_result(cursor, report_id, a)
            print("   Result:", a)
            passed.append(check("238000 /cumm bhi sahi classify (DB unit mein convert hua)",
                                a["classification"] == b["classification"]
                                and a["classification"] != "unit_mismatch"))

    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCENARIO 5 - Number-in-source check (bina kisi API ke)")
    print("=" * 60)
    src = [{"text": "<p width=100>Normal adult range: 30 - 100 ng/mL (75-250 nmol/L)</p>", "url": "t", "source": "t"}]
    ok, _ = verify_numbers_in_sources({"low": 30, "high": 100}, src)
    passed.append(check("source ke numbers (30-100) accept", ok))
    ok, _ = verify_numbers_in_sources({"low": 5, "high": 8}, src)
    passed.append(check("source mein na milne wale numbers (5-8) reject", not ok))
    ok, _ = verify_numbers_in_sources({"low": 150000, "high": 400000},
                                      [{"text": "Platelets 150,000 - 400,000 /uL", "url": "t", "source": "t"}])
    passed.append(check("150,000 jaise comma wale numbers sahi pehchane", ok))
    ok, _ = verify_numbers_in_sources({"low": 13.5, "high": 17.5},
                                      [{"text": "Hb males 13.5 to 17.5 g/dL", "url": "t", "source": "t"}])
    passed.append(check("decimal numbers (13.5-17.5) accept", ok))

    conn.commit()
    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print(f"FINAL: {sum(passed)}/{len(passed)} checks PASS")
    print("=" * 60)