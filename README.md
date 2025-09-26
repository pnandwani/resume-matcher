# Resume ⇄ Job Description Matcher (Local, No API)

This tool ranks your resumes against one or more job descriptions and surfaces the best match for each JD.

## Features
- Works with **PDF**, **DOCX**, or **TXT** files.
- Hybrid scoring: **TF–IDF cosine similarity** + **skill/keyword coverage**.
- Lightweight, **no external API** needed.
- Outputs a **ranked CSV per JD** and a **summary CSV** of top matches.

## Quick Start

1. Create two folders:

2. Install requirements (ideally in a virtualenv):

3. Run:
- `--alpha`: weight for TF–IDF cosine (default 0.6)
- `--beta`: weight for keyword/skill coverage (default 0.4)

4. See results in `out/`:
- `summary_top_matches.csv` — best resume per JD.
- `rankings/<JD_stem>__rankings.csv` — full ranking for that JD.

## How Scoring Works

**FinalScore = alpha * Cosine(TF–IDF) + beta * Coverage**

- **Cosine(TF–IDF)** compares the overall textual similarity.
- **Coverage** looks at how many important **skills/keywords** from the JD
  (built-in glossary + auto-extracted keywords) appear in the resume.

We also lightly **boost matches** found under common resume “Skills” sections.
