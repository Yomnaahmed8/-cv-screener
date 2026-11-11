# CV Screener

Ranks CVs against a job description in seconds, and shows which required skills each candidate has or is missing.

![Dashboard](screenshot.png)

## What it does
- Reads PDF CVs (upload your own, or use the sample CVs)
- Scores each CV with TF-IDF text similarity plus skill coverage
- Filters by minimum years of experience
- Lets you add extra required skills and tune the scoring weight
- Exports the ranking to CSV

## Run it
    pip install -r requirements.txt
    python make_cvs.py
    streamlit run app.py

## Note
This is a screening aid, not a hiring decision maker. The sample CVs are fictional. A person should always review the candidates.

## Built with
Python, Streamlit, scikit-learn, pandas, pdfplumber, Plotly