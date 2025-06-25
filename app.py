from flask import Flask, render_template, request, send_file, session
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = []
    return emails, names

@app.route('/', methods=['GET', 'POST'])
def index():
    massage = ''
    results = session.get('results', [])

    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')

        valid_resumes = []
        for i in resume_files:
            if i.filename.endswith('.pdf'):
                valid_resumes.append(i)
            else:
                massage = 'تم تجاهل بعض الملفات لأنها ليست بصيغة PDF'

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        processed_resumes = []
        for resume_file in valid_resumes:
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text))

        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

        ranked_resumes = []
        for (names, emails, resume_text) in processed_resumes:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
            ranked_resumes.append((names, emails, similarity))

        ranked_resumes.sort(key=lambda x: x[2], reverse=True)
        session['results'] = ranked_resumes
        results = ranked_resumes

    return render_template('index.html', results=results, massage=massage)

@app.route('/download_csv')
def download_csv():
    results = session.get('results', [])  # استرجاع النتائج من السيشن
    csv_content = "Rank,Email,Similarity\n"
    for rank, (names, emails, similarity) in enumerate(results, start=1):
        email = emails[0] if emails else "N/A"
        csv_content += f"{rank},{email},{similarity:.2f}\n"

    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_content)

    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")

if __name__ == '__main__':
    app.run(debug=True)
