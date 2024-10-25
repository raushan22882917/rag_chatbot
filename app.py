from flask import Flask, render_template, request, redirect, url_for, session, flash
import psycopg2
import os
from PyPDF2 import PdfReader, errors as pdf_errors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from werkzeug.security import generate_password_hash, check_password_hash
from pdf2image import convert_from_path
import pytesseract
import requests
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", 'wesrdlkijhgfdsdfghjklrd')  # Change to a random secret key

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        dbname="boarddata",
        user="root",
        password="jlilaCzBtCvke7PNN3XnAQCiSrw49b7q",  # Use a variable for the password
        host="dpg-csau4fggph6c73a6bgs0-a",
        port="5432"
    )
    return conn

# Admin email list
ADMIN_EMAILS = ["raushan22882917@gmail.com", "su-22016@sitare.org"]

# Function to extract text from PDFs with OCR fallback
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text() or ""
                    if not page_text:  # If no text is extracted, try OCR
                        images = convert_from_path(pdf, first_page=page.page_number + 1, last_page=page.page_number + 1)
                        for image in images:
                            page_text += pytesseract.image_to_string(image)
                    text += page_text
                except ValueError as ve:
                    print(f"ValueError encountered in {pdf} on page {page}: {ve}")
                    continue
        except (pdf_errors.PdfReadError, ValueError) as e:
            print(f"Error reading {pdf}: {e}")
            continue
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def clean_text(answer):
    """Removes unnecessary markdown symbols and formats steps into new lines."""
    cleaned_answer = re.sub(r'\*\*', '', answer)
    cleaned_answer = re.sub(r'(\*\*Step \d+)', r'\n\1', cleaned_answer)
    return cleaned_answer

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "Answer is not available in the context." Please do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and respond based on vector store and conversational chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Function to fetch image from Gemini/Google API
# Function to fetch image from Google Custom Search API
# Function to fetch image from Google API
def fetch_image(query):
    search_url = f"https://www.googleapis.com/customsearch/v1?key={google_api_key}&cx=YOUR_SEARCH_ENGINE_ID&q={query}&searchType=image&num=1"
    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                return data['items'][0]['link'], data['items'][0]['title']  # Return both image URL and title
    except Exception as e:
        print(f"Error fetching image: {e}")
    return None, None  # Return None if no image found

# Function to fetch top search links from Google
def fetch_links(query):
    # Prepare the query URL
    query = '+'.join(query.split())
    url = f'https://www.google.com/search?q={query}'
    
    # Send a request to Google
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    
    # Parse the HTML response
    soup = BeautifulSoup(response.text, 'html.parser')

    links = []
    for item in soup.select('h3'):
        parent = item.find_parent('a')
        if parent and parent['href']:
            link_title = item.get_text()
            link_url = parent['href']
            links.append({"title": link_title, "url": link_url})
            if len(links) >= 3:  # Limit to top 5 links
                break

    return links

# Index page route for regular users
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_email' in session and not session.get('is_admin', False):  # Check if regular user is logged in
        answer = None
        search_links = []  # For storing the top search links
        loading = False  # To show loading status
        
        if request.method == 'POST':
            user_question = request.form['user_question']
            loading = True  # Set loading to true while processing

            # Get the raw answer from the AI model
            raw_answer = user_input(user_question)
            answer = clean_text(raw_answer)  # Clean the answer before passing to template

            # Fetch the top 5 links related to the user question
            search_links = fetch_links(user_question)  # Fetch the links
            
            loading = False  # Set loading to false after processing
            
        return render_template('index.html', answer=answer, search_links=search_links, loading=loading)  # Pass answer and search_links to template
    else:
        flash('Please log in as a user to access this page.', 'warning')
        return redirect(url_for('login'))





    
@app.route('/about')
def about():
    return render_template('about.html')


# Admin login route
# Admin login route (no password required for admins)
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_email' in session:
        return redirect(url_for('admin'))

    if request.method == 'POST':
        email = request.form['admin_email']  # Use 'admin_email' here, not 'email'

        if email in ADMIN_EMAILS:
            session['admin_email'] = email
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin'))
        else:
            flash('You are not authorized to access this page.', 'danger')

    return render_template('admin_login.html')



# Admin Page
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'admin_email' not in session:
        flash('You must log in as admin to access this page.', 'warning')
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        if 'pdf_docs' in request.files:
            pdf_docs = request.files.getlist('pdf_docs')
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                flash('PDFs processed successfully!', 'success')
            else:
                flash('Please upload at least one PDF file.', 'warning')

    return render_template('admin.html')

# User login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT password FROM users WHERE email = %s', (email,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['user_email'] = email
            session['is_admin'] = False  # Set admin status to False for regular users
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

# Signup page route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            
            # Hash the password
            hashed_password = generate_password_hash(password)

            # Save the new user in the database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, hashed_password))
            conn.commit()
            cur.close()
            conn.close()

            flash('Signup successful!', 'success')
            return redirect(url_for('login'))
        except psycopg2.Error as e:
            flash(f'Database error: {e}', 'danger')
        except KeyError as e:
            flash(f'Missing field: {e}', 'danger')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')

    return render_template('signup.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('admin_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))




@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_email' not in session:
        flash('You need to be logged in to give feedback.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        question_1 = request.form['question_1']
        question_2 = request.form['question_2']
        question_3 = request.form['question_3']
        question_4 = request.form['question_4']
        comments = request.form['comments']
        user_email = session['user_email']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO feedback (user_email, question_1, question_2, question_3, question_4, comments) VALUES (%s, %s, %s, %s, %s, %s)',
                    (user_email, question_1, question_2, question_3, question_4, comments))
        conn.commit()
        cur.close()
        conn.close()

        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('index'))

    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
