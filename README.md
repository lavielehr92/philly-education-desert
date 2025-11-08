
# Philadelphia Education Desert Dashboard — Block-Group View

An interactive Streamlit app that maps an "Education Desert Index" across **Philadelphia block groups**, pins two Cornerstone addresses, and surfaces high-need areas for outreach.

## Features

- 📊 **Interactive Map**: Visualize Education Desert Index and other metrics across Philadelphia block groups
- 🎛️ **Customizable Views**: Select different metrics, color scales, opacity, and basemap styles
- 📈 **Comprehensive Metrics**: 
  - Education: % < HS, % Bachelor's+
  - Access: % No Vehicle, % No Internet
  - Demographics: % Children, % Poverty, % Unemployed, % Renter
  - Economic: Median Household Income
- 🏆 **Rankings**: Top 10 education desert block groups
- 📊 **Detailed Data**: Sortable table with all metrics
- ⬇️ **Download**: Export complete dataset as CSV
- ⭐ **Site Markers**: Cornerstone location pins on the map

## Quick start (local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Census API Key
1. Visit https://api.census.gov/data/key_signup.html
2. Register for a free API key
3. You'll receive the key via email

### 3. Configure your API key

Create a `.streamlit/secrets.toml` file in the project root:

```toml
CENSUS_API_KEY = "your_actual_api_key_here"
```

**Note**: The `.streamlit/secrets.toml` file is gitignored to protect your API key.

Alternatively, you can set it as an environment variable:
```bash
export CENSUS_API_KEY="your_actual_api_key_here"
```

### 4. Run the app
```bash
streamlit run education_desert_philly.py
```

The app will open in your browser at http://localhost:8501

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

- `education_desert_philly.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml` — Census API key configuration (template provided, add your own key)
- `README.md` — This file
- `.gitignore` — Protects sensitive files from being committed

## Data Sources & Methodology

### ACS Variables (Block-Group Level)

The app uses ACS 5-year **detailed tables** at **block-group** level:

**Education**:
- B15003: Educational attainment for population 25+

**Access**:
- B28002: Internet access
- B08201: Vehicle availability

**Demographics**:
- B09001: Population under 18
- B01003: Total population
- B17001: Poverty status
- B23025: Employment status
- B25003: Tenure (renter vs owner)

**Economic**:
- B19013: Median household income

### Education Desert Index (EDI)

The EDI combines three z-scored pillars, scaled 0-100:

1. **Need Score**: % < HS + % children
2. **Choice Gap Score**: % < HS - % Bachelor's+
3. **Access Friction Score**: % no vehicle + % no internet + (inverted median income)

Rankings are relative **within Philadelphia County** only.

### Geography

- Geometry fetched from **TIGERweb** for Philadelphia County block groups
- Joined on GEOID (12-digit: State + County + Tract + Block Group)
- Cornerstone sites geocoded via Census Geocoding API

## Troubleshooting

### "No Census API key found"
- Make sure you've created `.streamlit/secrets.toml` with your API key
- Or set the `CENSUS_API_KEY` environment variable
- Get a free key at https://api.census.gov/data/key_signup.html

### "ACS error 403" or rate limit errors
- The Census API has rate limits
- Wait a few minutes and try again
- If using an old key, generate a new one

### "Geometry match: X / Y block groups"
- This shows how many block groups successfully joined with map geometry
- If the match is low (<90%), there may be a GEOID mismatch
- The app attempts to fix this automatically

### Map not displaying all block groups
- Check the geometry match message on the map
- Verify your internet connection (geometry fetched from TIGERweb)
- Try refreshing the page

## Support

For issues or questions, please open an issue on the GitHub repository.

