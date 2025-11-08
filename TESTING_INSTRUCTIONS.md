# Setup and Testing Instructions

## For the User/Reviewer

This document provides step-by-step instructions to set up and test the enhanced Philadelphia Education Desert Dashboard.

## Prerequisites

1. **Python 3.11** (as specified in runtime.txt)
2. **Census API Key** (free, required for data access)

## Step 1: Get a Census API Key

1. Visit: https://api.census.gov/data/key_signup.html
2. Fill out the registration form with:
   - Organization name
   - Email address
   - Agree to terms of service
3. Click "Submit"
4. Check your email for the API key (usually arrives within minutes)
5. Save your API key - you'll need it in the next step

## Step 2: Configure the API Key

### Option A: Using secrets.toml (Recommended)

1. The repository already has a template file at `.streamlit/secrets.toml`
2. Edit this file and replace `YOUR_CENSUS_API_KEY_HERE` with your actual key:

```bash
# Open the file in your editor
nano .streamlit/secrets.toml

# Or use sed to replace directly
sed -i 's/YOUR_CENSUS_API_KEY_HERE/your_actual_key_here/' .streamlit/secrets.toml
```

The file should look like:
```toml
CENSUS_API_KEY = "abc123def456ghi789jkl012mno345pqr678"
```

### Option B: Using Environment Variable

Alternatively, export the key as an environment variable:

```bash
export CENSUS_API_KEY="your_actual_key_here"
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- streamlit>=1.36
- pandas>=2.0
- numpy>=1.24
- requests>=2.31
- plotly>=5.22

## Step 4: Run the Application

```bash
streamlit run education_desert_philly.py
```

The application will:
1. Start a local web server
2. Open your browser automatically to http://localhost:8501
3. Display the dashboard

## Step 5: Verify Functionality

### ✅ Initial Load
- [ ] Page loads without errors
- [ ] Green message: "✅ Census API key loaded from secrets"
- [ ] Spinner shows "Loading ACS data..."
- [ ] Data loads successfully (takes 10-30 seconds)

### ✅ Map View Tab
- [ ] Interactive map displays
- [ ] Philadelphia block groups are colored
- [ ] Geometry match message shows high percentage (e.g., "380 / 380")
- [ ] Two gold star markers appear (Cornerstone sites)
- [ ] Hover over block groups shows detailed tooltip with all metrics
- [ ] Tooltip includes: Education, Access, Demographics sections

### ✅ Sidebar Controls
Test each control to ensure it updates the map:

- [ ] **Map Metric**: Change selection → map colors update
  - Try: Education Desert Index, % Poverty, % Unemployed, % Renter
- [ ] **Color Scale**: Change selection → colors change
  - Try: YlOrRd, Blues, Viridis
- [ ] **Opacity**: Move slider → map transparency changes
  - Try: 0.3, 0.7, 1.0
- [ ] **Basemap Style**: Change selection → background changes
  - Try: none, carto-positron
- [ ] **EDI Tier Filter**: Uncheck options → map filters
  - Try: Show only "Higher" tier

### ✅ Rankings Tab
- [ ] Four metric cards display:
  - Total Block Groups
  - Avg Education Desert Index
  - High-Need Areas
  - Avg Poverty Rate
- [ ] Top 10 table displays with all metrics
- [ ] Table includes new metrics: % Poverty, % Unemployed, % Renter
- [ ] Values are properly formatted (1 decimal place)

### ✅ Detailed Data Tab
- [ ] Full data table displays
- [ ] Caption shows count of filtered block groups
- [ ] Table is sortable (click column headers)
- [ ] All metrics are present
- [ ] Scrollbar works (table is tall)

### ✅ Download Tab
- [ ] Download button is visible
- [ ] Clicking downloads a CSV file
- [ ] CSV contains all metrics
- [ ] Filename: `philly_education_desert_blockgroups.csv`

### ✅ Methodology Section
- [ ] Expandable section at bottom
- [ ] Click to expand → methodology displays
- [ ] Includes all three pillars: Need, Choice Gap, Access Friction
- [ ] Lists all additional metrics
- [ ] Shows data sources

## Step 6: Take Screenshots

For documentation purposes, please take screenshots of:

1. **Map View** - showing the full interface with sidebar controls
2. **Rankings Tab** - showing the highlight cards and top 10 table
3. **Detailed Data Tab** - showing the complete data table
4. **Hover Tooltip** - showing the enhanced tooltip with all metrics

## Expected Results

### Data Completeness
- **Block Groups**: ~380 for Philadelphia County
- **Geometry Match**: Should be 100% or very close (e.g., 380/380)
- **Metrics**: All 10+ metrics should have values (some may be NaN for certain block groups)

### New Metrics
The following new metrics should be visible throughout:
- **% Poverty**: Percentage below poverty line
- **% Unemployed**: Percentage of labor force unemployed
- **% Renter**: Percentage of renter-occupied housing

### Performance
- **Initial Load**: 10-30 seconds (downloads data from Census API)
- **Subsequent Loads**: Faster (data is cached for 24 hours)
- **UI Interactions**: Immediate (all controls update map instantly)

## Troubleshooting

### Problem: "❌ No Census API key found"
**Solution**: 
- Verify `.streamlit/secrets.toml` exists and has your key
- Check the file isn't named `secrets.toml.template`
- Try setting environment variable instead

### Problem: "❌ Data load failed: ACS error 403"
**Solution**:
- Your API key might be invalid
- Get a new key from https://api.census.gov/data/key_signup.html
- Check for typos in your key

### Problem: "Geometry match: 200 / 380 block groups"
**Solution**:
- This indicates GEOID mismatch (should be rare)
- Check internet connection (TIGERweb geometry fetch)
- Try refreshing the page
- If persistent, check Census API status

### Problem: Map displays but some areas are gray
**Solution**:
- Gray areas have no data in ACS (normal for some block groups)
- Try filtering by EDI tier to see only areas with data
- Check the detailed data tab to see which block groups have data

### Problem: Rate limit error
**Solution**:
- Census API has rate limits (~500 requests/day per IP)
- Data is cached for 24 hours, so refresh shouldn't cause issues
- Wait a few minutes and try again
- If developing, use cached data from previous successful load

## Deployment Options

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Add API key in App Settings → Secrets
5. Deploy!

### Hugging Face Spaces (Free)
1. Create Space at https://huggingface.co/spaces
2. Select Streamlit as framework
3. Upload files
4. Add API key in Settings → Repository Secrets
5. Space auto-deploys!

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review IMPLEMENTATION_SUMMARY.md for technical details
3. Check Streamlit logs in terminal for detailed errors
4. Verify all dependencies installed correctly
5. Try in a fresh virtual environment

## Next Steps After Verification

Once testing is complete:
1. Take screenshots for documentation
2. Note any issues or suggestions
3. Consider deploying to Streamlit Cloud for team access
4. Share the public URL with stakeholders

---

**Happy Testing!** 🎉
