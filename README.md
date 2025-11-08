
# Philadelphia Education Desert Dashboard — Block-Group View

An interactive Streamlit app that maps an "Education Desert Index" across **Philadelphia block groups**, pins two Cornerstone addresses, and surfaces high-need areas for outreach.

## Quick start (local)
```bash
pip install -r requirements.txt
streamlit run education_desert_philly.py
```

## Streamlit Community Cloud (shareable link)
1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io and deploy.
3. In **App settings → Secrets**, add:
```
CENSUS_API_KEY = "YOUR_KEY_HERE"
```
4. Click your public app URL and share it with the team.

## Hugging Face Spaces (alternative)
1. Create a new **Space** with type **Streamlit**.
2. Upload all files in this folder.
3. In **Settings → Repository secrets**, add `CENSUS_API_KEY` with your key.

## Files
- `education_desert_philly.py` — Streamlit app
- `requirements.txt` — Python deps
- `.streamlit/secrets.toml` — place to put your Census API key (optional locally; required for cloud)
- `README.md` — these instructions

## Notes
- App uses ACS 5-year **detailed tables** at **block-group** level:
  - B15003 (educational attainment 25+), B28002 (internet), B08201 (vehicles), B09001 (children), B01003 (total pop), B19013 (income).
- Geometry is fetched from **TIGERweb** for Philadelphia County block groups and joined on `GEOID`.
- The index is relative **within Philadelphia** (not comparing to other counties).
