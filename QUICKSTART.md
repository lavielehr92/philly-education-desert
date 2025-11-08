# 🚀 Quick Start Guide

Get your Philadelphia Education Desert Dashboard running in **3 minutes**!

## Step 1: Get a FREE Census API Key (2 minutes)

1. Visit: **https://api.census.gov/data/key_signup.html**
2. Fill out the simple form (name, email, organization)
3. Check your email for the API key (arrives instantly)
4. Copy your key (looks like: `abc123def456ghi789...`)

## Step 2: Configure Your Key (30 seconds)

### Option A: Create the secrets file (Recommended)

```bash
# Create the directory and file
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Edit the file and paste your API key
nano .streamlit/secrets.toml
```

Replace `YOUR_CENSUS_API_KEY_HERE` with your actual key:
```toml
CENSUS_API_KEY = "paste_your_key_here"
```

### Option B: Use environment variable

```bash
export CENSUS_API_KEY="paste_your_key_here"
```

## Step 3: Install and Run (30 seconds)

```bash
# Install dependencies (one-time)
pip install -r requirements.txt

# Run the dashboard
streamlit run education_desert_philly.py
```

That's it! Your browser will automatically open to **http://localhost:8501** 🎉

---

## What You'll See

- **📍 Map View**: Interactive map of Philadelphia showing education desert areas
- **🏆 Rankings**: Top education desert block groups
- **📊 Data**: Complete dataset with all metrics
- **⬇️ Download**: Export data as CSV

## Need Help?

- **Dashboard won't start?** Make sure you completed Step 2 (API key setup)
- **"No Census API key found" error?** Double-check your `.streamlit/secrets.toml` file
- **Data loading slowly?** First load takes 10-30 seconds (fetching Census data)
- **More questions?** See `README.md` or `TESTING_INSTRUCTIONS.md`

## Deploy Online (Optional)

Want to share with your team? Deploy for FREE to Streamlit Cloud:

1. Push this code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Add your API key in **App Settings → Secrets**
5. Get a public URL to share!

---

**Happy Analyzing!** 📊✨
