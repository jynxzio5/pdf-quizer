from flask import Flask, render_template, request, jsonify
import pdfplumber
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import datetime
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename
import logging

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found in environment variables")

# التعامل مع الأخطاء
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'حدث خطأ داخلي في الخادم', 'details': str(error)}), 500

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"Page not found: {str(error)}")
    return jsonify({'error': 'الصفحة غير موجودة'}), 404

def extract_text_from_pdf(file_path):
    """استخراج النص من ملف PDF."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"خطأ في استخراج النص: {str(e)}")
        return None

def generate_questions_with_openai(text, num_questions=5, question_type='multiple_choice'):
    """توليد أسئلة باستخدام OpenAI."""
    try:
        if question_type == 'multiple_choice':
            prompt = f"""Generate {num_questions} multiple choice questions in Arabic based on this text:

{text}

Format each question like this:
السؤال: [Question text]
أ) [Option A]
ب) [Option B]
ج) [Option C]
د) [Option D]
الإجابة الصحيحة: [Correct option letter]"""

        elif question_type == 'essay':
            prompt = f"""Generate {num_questions} essay questions in Arabic based on this text:

{text}

Format each question like this:
السؤال: [Question text]
إرشادات للإجابة: [Guidelines for answering]"""

        elif question_type == 'flashcards':
            prompt = f"""Generate {num_questions} flashcards in Arabic based on this text:

{text}

Format each flashcard like this:
السؤال: [Front of card]
الإجابة: [Back of card]"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد متخصص في توليد أسئلة تعليمية باللغة العربية."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"خطأ في توليد الأسئلة: {str(e)}")
        return None

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return jsonify({'error': 'حدث خطأ في عرض الصفحة الرئيسية'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """معالجة رفع الملف وتوليد الأسئلة."""
    try:
        # التحقق من وجود الملف
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'لم يتم تحميل ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not file.filename.endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'error': 'يجب رفع ملف PDF فقط'}), 400

        # التحقق من معلمات الطلب
        question_type = request.form.get('type')
        if not question_type:
            logger.error("No question type specified")
            return jsonify({'error': 'يجب تحديد نوع الأسئلة'}), 400
            
        try:
            num_questions = int(request.form.get('count', 5))
        except ValueError:
            logger.error("Invalid question count")
            return jsonify({'error': 'عدد الأسئلة غير صالح'}), 400

        # حفظ الملف
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        # استخراج النص
        text = extract_text_from_pdf(filepath)
        if not text:
            logger.error("Failed to extract text from PDF")
            return jsonify({'error': 'فشل في استخراج النص من الملف'}), 400

        # توليد الأسئلة
        questions = generate_questions_with_openai(text, num_questions, question_type)
        if not questions:
            logger.error("Failed to generate questions")
            return jsonify({'error': 'فشل في توليد الأسئلة'}), 500

        # حذف الملف المؤقت
        try:
            os.remove(filepath)
            logger.info(f"Temporary file removed: {filepath}")
        except Exception as e:
            logger.error(f"Failed to remove temporary file: {str(e)}")

        return jsonify({
            'success': True,
            'questions': questions
        })

    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)  # تعطيل وضع التصحيح في الإنتاج
else:
    # تهيئة التطبيق للإنتاج
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
