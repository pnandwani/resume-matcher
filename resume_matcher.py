import argparse
import os
import re
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- TEXT EXTRACTION HELPERS ----------

def read_txt(path: str) -> str:
    """Read text file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    """Read .docx file."""
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: str) -> str:
    """Read text from a PDF file."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        raise RuntimeError("pdfminer.six is required for PDF parsing. Install with: pip install pdfminer.six")
    return extract_text(path)

def extract_text_auto(path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = pathlib.Path(path).suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".pdf":
        return read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {ext} for {path}")

# ---------- TEXT NORMALIZATION ----------

NONWORD = re.compile(r"[^a-z0-9#+]")
SPACE = re.compile(r"\s+")

def normalize(text: str) -> str:
    """Lowercase, remove special chars, collapse spaces."""
    text = text.lower()
    text = NONWORD.sub(" ", text)
    text = SPACE.sub(" ", text).strip()
    return text

# ---------- SKILLS SECTION BOOST ----------

SKILLS_SECTION_RE = re.compile(r"(skills|technical skills|skills & tools|tech skills)[:\n]", re.I)

def extract_skills_section(text: str) -> str:
    """Extract ~800 chars after a Skills section heading, if present."""
    m = SKILLS_SECTION_RE.search(text)
    if not m:
        return ""
    start = m.end()
    end = min(len(text), start + 800)
    return text[start:end]

# ---------- BUILT-IN SKILL GLOSSARY ----------

TECH_SKILLS = {
    # languages
    "python","java","kotlin","c","c++","c#","go","golang","javascript","typescript","ruby","scala","swift","matlab","r",
    # web/tools
    "html","css","react","angular","vue","node","node.js","next.js","spring","spring boot","django","flask","fastapi",
    # cloud/devops
    "aws","azure","gcp","docker","kubernetes","k8s","terraform","ansible","jenkins","gitlab","github actions","argo",
    # data
    "sql","nosql","postgres","mysql","sqlite","oracle","mongodb","dynamodb","redis","elasticsearch","spark","hadoop",
    "kafka","kinesis","pandas","numpy","scikit-learn","airflow","dbt",
    # testing
    "selenium","pytest","junit","testng","cypress","playwright","postman","rest-assured","jmeter","locust",
    # platforms
    "linux","unix","macos","windows",
    # security/infra
    "oauth","oidc","jwt","iam","vpc","ec2","s3","lambda","cloudformation","cloudwatch","prometheus","grafana",
    # misc
    "rest","grpc","graphql","microservices","ci/cd","event-driven","message queue","pubsub",
}

# ---------- KEYWORD EXTRACTION FROM JD ----------

def auto_keywords_from_jd(jd_text: str, top_k: int = 40) -> List[str]:
    """Auto-extract top keywords (unigrams + bigrams) from a job description."""
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english", max_features=5000)
    X = vect.fit_transform([jd_text])
    weights = X.toarray()[0]
    indices = np.argsort(weights)[::-1]
    feats = np.array(vect.get_feature_names_out())[indices]
    
    filt = []
    for tok in feats:
        if re.fullmatch(r"[a-z0-9][a-z0-9+.#\- ]{1,24}", tok):
#                               ^ single backslash before hyphen

            if len(tok) >= 2:
                filt.append(tok.strip())
        if len(filt) >= top_k:
            break
    return filt

def build_target_terms(jd_text: str) -> List[str]:
    """Combine built-in skills and auto-extracted JD keywords."""
    base = set([t.lower() for t in TECH_SKILLS])
    auto = auto_keywords_from_jd(jd_text, top_k=60)
    for t in auto:
        base.add(t.lower())
    return sorted(base)
# ---------- COVERAGE SCORING ----------

def coverage_score(
    resume_text: str,
    target_terms: List[str],
    skills_section: str = "",
    boost: float = 0.06
) -> Tuple[float, int, int]:
    """
    Compute coverage of JD target terms in the resume.
    - Base coverage = hits / total terms
    - Small incremental boost for terms found inside a 'Skills' section
    Returns: (coverage_score, hits, total_terms)
    """
    total = 0
    hits = 0
    bonus_hits = 0

    norm_resume = " " + normalize(resume_text) + " "
    norm_skills = " " + normalize(skills_section) + " " if skills_section else ""

    for term in target_terms:
        total += 1
        term_norm = " " + normalize(term) + " "
        if term_norm in norm_resume:
            hits += 1
            if norm_skills and (term_norm in norm_skills):
                bonus_hits += 1

    if total == 0:
        return 0.0, 0, 0

    base_cov = hits / total
    # incremental boost per skill appearing inside the Skills section; lightly capped
    cov = min(1.0, base_cov + boost * bonus_hits)
    return cov, hits, total


# ---------- RANKING PIPELINE ----------

def rank_resumes_for_jd(
    jd_text: str,
    resume_texts: Dict[str, str],
    alpha: float,
    beta: float
) -> pd.DataFrame:
    """
    Build TF–IDF over JD + all resumes, compute cosine sim and coverage, and
    combine into final score.
    FinalScore = alpha * cosine + beta * coverage
    """
    # Vectorize (unigrams + bigrams) for semantic-ish matching
    docs = [jd_text] + [resume_texts[p] for p in resume_texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000)
    X = vectorizer.fit_transform(docs)
    jd_vec = X[0:1]
    res_mat = X[1:]

    cosine = cosine_similarity(jd_vec, res_mat)[0]

    # Coverage against built-in + auto JD keywords
    target_terms = build_target_terms(jd_text)
    cov_scores, cov_hits, cov_totals = [], [], []
    for path, txt in resume_texts.items():
        skills_sec = extract_skills_section(txt)
        cov, hits, total = coverage_score(txt, target_terms, skills_section=skills_sec, boost=0.06)
        cov_scores.append(cov)
        cov_hits.append(hits)
        cov_totals.append(total)

    cosine = np.array(cosine)
    cov = np.array(cov_scores)

    rows = []
    for (path, _), c, v, h, t in zip(resume_texts.items(), cosine, cov, cov_hits, cov_totals):
        rows.append({
            "resume_path": path,
            "cosine_similarity": round(float(c), 4),
            "coverage_score": round(float(v), 4),
            "coverage_hits": int(h),
            "coverage_total_terms": int(t),
            "final_score": round(float(alpha * c + beta * v), 4),
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df


# ---------- IO HELPERS ----------

def collect_texts_from_folder(folder: str) -> Dict[str, str]:
    """
    Recursively read .pdf/.docx/.txt under a folder into {path: raw_text}.
    If a file fails to parse, it is included with empty text so the run continues.
    """
    texts = {}
    for root, _, files in os.walk(folder):
        for f in files:
            ext = pathlib.Path(f).suffix.lower()
            if ext not in {".pdf", ".docx", ".txt"}:
                continue
            path = os.path.join(root, f)
            try:
                raw = extract_text_auto(path)
                texts[path] = raw or ""
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
                texts[path] = ""
    return texts


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Rank resumes against one or more JDs and output CSVs."
    )
    ap.add_argument("--jds", required=True, help="Folder with JD files (.pdf/.docx/.txt)")
    ap.add_argument("--resumes", required=True, help="Folder with resumes (.pdf/.docx/.txt)")
    ap.add_argument("--out", default="out", help="Output folder (default: out)")
    ap.add_argument("--alpha", type=float, default=0.6, help="Weight for cosine similarity (default: 0.6)")
    ap.add_argument("--beta", type=float, default=0.4, help="Weight for keyword coverage (default: 0.4)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "rankings"), exist_ok=True)

    jd_texts = collect_texts_from_folder(args.jds)
    resume_texts = collect_texts_from_folder(args.resumes)

    if not jd_texts:
        raise SystemExit(f"No JDs found in {args.jds}. Supported: .pdf .docx .txt")
    if not resume_texts:
        raise SystemExit(f"No resumes found in {args.resumes}. Supported: .pdf .docx .txt")

    summary_rows = []
    for jd_path, jd_raw in jd_texts.items():
        if not jd_raw.strip():
            print(f"[WARN] JD appears empty or unreadable: {jd_path}")
            continue

        print(f"[INFO] Ranking resumes for JD: {jd_path}")
        df = rank_resumes_for_jd(jd_raw, resume_texts, alpha=args.alpha, beta=args.beta)

        stem = pathlib.Path(jd_path).stem
        out_csv = os.path.join(args.out, "rankings", f"{stem}__rankings.csv")
        df.to_csv(out_csv, index=False)

        # Best match for this JD
        top = df.iloc[0].to_dict()
        top["jd_path"] = jd_path
        summary_rows.append(top)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("final_score", ascending=False).reset_index(drop=True)
        summary_csv = os.path.join(args.out, "summary_top_matches.csv")
        summary_df.to_csv(summary_csv, index=False)

        with open(os.path.join(args.out, "summary_top_matches.json"), "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        print(f"[DONE] Wrote summary and rankings to: {args.out}")
    else:
        print("[WARN] No valid JDs processed. Nothing to summarize.")


if __name__ == "__main__":
    main()
