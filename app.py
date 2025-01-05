from flask import Flask, render_template, request, jsonify
import pdfplumber
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import uuid
import json
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename

try:
    # Initialize Firebase with explicit path
    service_account_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serviceAccountKey.json')
    
    # Load and validate the service account file
    with open(service_account_path, 'r') as f:
        service_account_info = json.load(f)
    
    cred = credentials.Certificate(service_account_info)
    
    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    raise e

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# تكوين Flask للـ production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'  # استخدام مجلد tmp في Vercel

# تكوين OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# التعامل مع الأخطاء
@app.errorhandler(500)
def internal_error(error):
    print(f"Internal error: {str(error)}")  # إضافة سجل للخطأ
    return jsonify({'error': 'حدث خطأ داخلي في الخادم'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'الصفحة غير موجودة'}), 404

def extract_text_from_pdf(file_path):
    """استخراج النص من ملف PDF."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"خطأ في استخراج النص: {str(e)}")
        return None

def generate_questions_with_openai(text, num_questions=5, question_type='multiple_choice'):
    """توليد أسئلة باستخدام OpenAI."""
    try:
        # تحضير التعليمات حسب نوع السؤال
        if question_type == 'multiple_choice':
            prompt = f"""قم بإنشاء {num_questions} أسئلة اختيار من متعدد باللغة العربية من النص التالي. 
            لكل سؤال، قم بتوفير 4 خيارات والإجابة الصحيحة.
            النص: {text[:4000]}  # تحديد حجم النص لتجنب تجاوز حد الرمز

            قم بتنسيق الإجابة بالشكل التالي:
            السؤال 1: [نص السؤال]
            أ) [الخيار الأول]
            ب) [الخيار الثاني]
            ج) [الخيار الثالث]
            د) [الخيار الرابع]
            الإجابة الصحيحة: [الحرف]

            """
        elif question_type == 'essay':
            prompt = f"""قم بإنشاء {num_questions} أسئلة مقالية باللغة العربية من النص التالي.
            لكل سؤال، قم بتوفير إجابة نموذجية.
            النص: {text[:4000]}

            قم بتنسيق الإجابة بالشكل التالي:
            السؤال 1: [نص السؤال]
            الإجابة النموذجية: [الإجابة]

            """
        else:  # flashcards
            prompt = f"""قم بإنشاء {num_questions} بطاقات تعليمية باللغة العربية من النص التالي.
            لكل بطاقة، قم بتوفير سؤال وإجابة.
            النص: {text[:4000]}

            قم بتنسيق الإجابة بالشكل التالي:
            البطاقة 1:
            السؤال: [نص السؤال]
            الإجابة: [الإجابة]

            """

        # استدعاء OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت معلم خبير في إنشاء أسئلة تعليمية عالية الجودة."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"خطأ في توليد الأسئلة: {str(e)}")
        return None

def save_to_firebase(questions, original_filename):
    try:
        # Create a new document with a unique ID
        doc_id = str(uuid.uuid4())
        doc_ref = db.collection('questions').document(doc_id)
        
        # Convert datetime to string to make it JSON serializable
        current_time = datetime.datetime.now().isoformat()
        
        # Prepare the document data
        doc_data = {
            'id': doc_id,
            'questions': questions,
            'original_filename': original_filename,
            'created_at': current_time,
            'updated_at': current_time
        }
        
        # Save to Firestore
        doc_ref.set(doc_data)
        return doc_id
    except Exception as e:
        print(f"Error saving to Firebase: {str(e)}")
        return None

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/auth/callback')
def auth_callback():
    return jsonify({'status': 'success'})

@app.route('/questions/history', methods=['GET'])
def get_questions_history():
    try:
        # Get all documents from the questions collection
        docs = db.collection('questions').stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                'id': data['id'],
                'original_filename': data['original_filename'],
                'created_at': data['created_at']
            })
        
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error getting history: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء تحميل السجل'}), 500

@app.route('/questions/<document_id>', methods=['GET'])
def get_questions(document_id):
    try:
        doc_ref = db.collection('questions').document(document_id)
        doc = doc_ref.get()
        
        if doc.exists:
            return jsonify(doc.to_dict())
        else:
            return jsonify({'error': 'لم يتم العثور على الأسئلة'}), 404
    except Exception as e:
        print(f"Error getting questions: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء استرجاع البيانات'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """معالجة رفع الملف وتوليد الأسئلة."""
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم تحميل ملف'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'يجب رفع ملف PDF فقط'}), 400

    try:
        # حفظ الملف
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # استخراج النص
        text = extract_text_from_pdf(filepath)
        if not text:
            return jsonify({'error': 'فشل في استخراج النص من الملف'}), 400

        # الحصول على معلمات الطلب
        question_type = request.form.get('type', 'multiple_choice')
        num_questions = int(request.form.get('count', 5))

        # توليد الأسئلة باستخدام OpenAI
        questions = generate_questions_with_openai(text, num_questions, question_type)
        if not questions:
            return jsonify({'error': 'فشل في توليد الأسئلة'}), 500

        # حفظ في Firebase
        document_id = save_to_firebase(questions, file.filename)

        # حذف الملف المؤقت
        os.remove(filepath)

        return jsonify({
            'success': True,
            'questions': questions,
            'document_id': document_id
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)  # تعطيل وضع التصحيح في الإنتاج
else:
    # تهيئة التطبيق للإنتاج
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
