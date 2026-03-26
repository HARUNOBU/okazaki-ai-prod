python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:OPENAI_API_KEY = ""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"


python scripts/build_docs_from_pdfs.py

python scripts/ingest_chroma.py

streamlit run app/app.py

 api_key="sk-..." 
 
  $env:OPENAI_API_KEY="sk-..."

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