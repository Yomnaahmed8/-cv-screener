import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

os.makedirs("cvs", exist_ok=True)

people = [
    ("Sara Hassan", "Data Analyst", 3,
     "Python, SQL, Excel, Power BI, pandas, data visualization",
     "Builds dashboards and reports that help managers make decisions."),
    ("Omar Khaled", "Machine Learning Engineer", 4,
     "Python, scikit-learn, TensorFlow, machine learning, NLP, pandas",
     "Builds and deploys machine learning models for prediction."),
    ("Mona Adel", "Frontend Developer", 2,
     "JavaScript, React, HTML, CSS, Figma, responsive design",
     "Creates fast and responsive web interfaces."),
    ("Youssef Ali", "Backend Developer", 5,
     "Python, Django, PostgreSQL, Docker, REST APIs, SQL",
     "Designs APIs and databases for web applications."),
    ("Nour Samir", "Accountant", 6,
     "Excel, QuickBooks, financial reporting, auditing, budgeting",
     "Prepares monthly financial statements and audits."),
    ("Hana Mostafa", "Data Scientist", 3,
     "Python, machine learning, SQL, statistics, pandas, Tableau",
     "Analyzes data and builds predictive models for business teams."),
    ("Karim Fathy", "Marketing Specialist", 4,
     "SEO, Google Ads, content writing, social media, analytics",
     "Runs online campaigns and grows brand awareness."),
    ("Laila Nabil", "AI Engineer", 2,
     "Python, NLP, deep learning, PyTorch, data analysis, machine learning",
     "Develops language and vision models for small business tools."),
]

for name, title, years, skills, summary in people:
    c = canvas.Canvas(f"cvs/{name.replace(' ', '_')}.pdf", pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 780, name)
    c.setFont("Helvetica", 13)
    c.drawString(50, 755, title)
    c.setFont("Helvetica", 11)
    c.drawString(50, 720, f"Experience: {years} years")
    c.drawString(50, 700, f"Skills: {skills}")
    c.drawString(50, 680, f"Summary: {summary}")
    c.save()

print("Done: 8 CVs in the cvs folder")