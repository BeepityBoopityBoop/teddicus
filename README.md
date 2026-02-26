# 📚 ITEC 3310 RAG Course Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Claude (Anthropic), deployed on Streamlit Community Cloud. Built as part of the Data Literacy & RAG Workshop — Module 3, Track C.

---

## What It Does

- Indexes the ITEC 3310 course syllabus into a vector store at startup
- Accepts student questions via a chat interface
- Retrieves the most relevant syllabus chunks and passes them to Claude
- Returns grounded, cited answers — and declines questions outside the document

**Pipeline:** `TextLoader → RecursiveCharacterTextSplitter → HuggingFace Embeddings (all-MiniLM-L6-v2) → ChromaDB → Claude claude-haiku-4-5-20251001`

---

## Deploy to Streamlit Community Cloud (Step-by-Step)

### Prerequisites
- A [GitHub](https://github.com) account (free)
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free, sign in with GitHub)
- An [Anthropic API key](https://console.anthropic.com) (get one at console.anthropic.com)

---

### Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it: `itec3310-rag-assistant` (or anything you like)
3. Set it to **Public** (required for Streamlit Community Cloud free tier)
4. Click **Create repository**

---

### Step 2 — Upload the Files

You have two options:

**Option A — GitHub web UI (no terminal needed):**
1. In your new repo, click **Add file → Upload files**
2. Upload ALL files from this folder, maintaining the folder structure:
   ```
   app.py
   requirements.txt
   .gitignore
   docs/
     itec3310_syllabus.txt
   .streamlit/
     config.toml
     secrets.toml.template   ← upload this, but DO NOT upload secrets.toml
   ```
3. Click **Commit changes**

**Option B — Git command line:**
```bash
cd rag-chatbot
git init
git add .
git commit -m "Initial commit — ITEC 3310 RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/itec3310-rag-assistant.git
git push -u origin main
```

---

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/itec3310-rag-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings**
5. In the **Secrets** box, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
   ```
6. Click **Deploy**

Streamlit will install dependencies and launch your app. First deploy takes ~3–5 minutes (downloading the sentence-transformers model). Your app will be live at:
```
https://YOUR_USERNAME-itec3310-rag-assistant-app-XXXX.streamlit.app
```

---

### Step 4 — Share Your Demo

Copy the URL from the Streamlit dashboard and share it. Anyone with the link can use the chatbot — no login required.

---

## Running Locally (Optional)

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/itec3310-rag-assistant.git
cd itec3310-rag-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and replace the placeholder with your real key

# 5. Run
streamlit run app.py
```

---

## Customising the Document

To swap in your own syllabus or policy document:
1. Replace `docs/itec3310_syllabus.txt` with your own `.txt` file
2. Update the `doc_path` line in `app.py` to match your filename
3. Update the hero title and subtitle in the `st.markdown` hero block
4. Commit and push — Streamlit Cloud will redeploy automatically

---

## File Structure

```
rag-chatbot/
├── app.py                          # Main Streamlit app + RAG pipeline
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes secrets.toml and caches
├── docs/
│   └── itec3310_syllabus.txt       # Knowledge base document
└── .streamlit/
    ├── config.toml                 # Dark theme + server config
    └── secrets.toml.template       # Secret key template (safe to commit)
```

---

## Architecture Notes

| Component | Choice | Reason |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Free, no API key, runs on CPU |
| Vector store | ChromaDB (in-memory) | No setup, resets on each deploy (fine for demo) |
| LLM | Claude claude-haiku-4-5-20251001 | Fast, low cost, strong instruction following |
| Chunk size | 400 tokens / 60 overlap | Balances context and retrieval precision |
| k (retrieved chunks) | 4 | Covers multi-section queries without exceeding context |

---

## Cost Estimate

Claude Haiku is extremely affordable. At typical student query lengths (~100 tokens input, ~200 output), you'd need roughly **5,000 queries** to spend $1.00. A demo with 50 visitors asking 10 questions each costs under $0.01.

---

*Built for the Data Literacy & RAG Workshop — Module 3, Track C · 2026*
