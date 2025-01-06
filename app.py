from flask import Flask, render_template, request, jsonify
import pdfplumber
import os
import datetime
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import firebase_admin
from firebase_admin import credentials, firestore, auth
import uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة نموذج T5
try:
    model_name = "google/mt5-small"  # نموذج أخف
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    model.eval()  # وضع التقييم
    t5_enabled = True
    logger.info("T5 model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading T5 model: {str(e)}")
    t5_enabled = False

# تهيئة Firebase
try:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
        "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n') if os.getenv('FIREBASE_PRIVATE_KEY') else None,
        "client_email": os.getenv('FIREBASE_CLIENT_EMAIL')
    })
    
    if not all([os.getenv('FIREBASE_PROJECT_ID'), os.getenv('FIREBASE_PRIVATE_KEY'), os.getenv('FIREBASE_CLIENT_EMAIL')]):
        logger.warning("Firebase credentials not found in environment variables. Authentication will be disabled.")
        firebase_enabled = False
    else:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_enabled = True
        logger.info("Firebase initialized successfully!")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    firebase_enabled = False

app = Flask(__name__)
CORS(app)

# تكوين Flask للـ production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp'  # استخدام مجلد tmp في Vercel

# التعامل مع الأخطاء
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'حدث خطأ داخلي في الخادم', 'details': str(error)}), 500

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"Page not found: {str(error)}")
    return jsonify({'error': 'الصفحة غير موجودة'}), 404

def verify_firebase_token(id_token):
    """التحقق من صحة رمز المصادقة."""
    try:
        if not firebase_enabled:
            return None
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return None

def save_to_firebase(user_id, questions, original_filename):
    """حفظ الأسئلة في Firebase."""
    try:
        if not firebase_enabled:
            logger.warning("Firebase is disabled. Questions will not be saved.")
            return None

        doc_id = str(uuid.uuid4())
        doc_data = {
            'user_id': user_id,
            'questions': questions,
            'filename': original_filename,
            'timestamp': datetime.datetime.utcnow().isoformat(),
        }
        
        doc_ref = db.collection('questions').document(doc_id)
        doc_ref.set(doc_data)
        logger.info(f"Questions saved to Firebase with ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Error saving to Firebase: {str(e)}")
        return None

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

def generate_questions_with_t5(text, num_questions=5, question_type='multiple_choice'):
    """توليد أسئلة باستخدام نموذج T5."""
    try:
        if not t5_enabled:
            logger.error("T5 model is not initialized")
            return None

        # تقسيم النص إلى أجزاء أصغر إذا كان طويلاً
        max_length = 512
        text_parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_questions = []
        for part in text_parts:
            # إعداد المدخلات حسب نوع السؤال
            if question_type == 'multiple_choice':
                input_text = f"generate multiple choice question: {part}"
            elif question_type == 'essay':
                input_text = f"generate essay question: {part}"
            else:  # flashcards
                input_text = f"generate flashcard: {part}"

            # ترميز النص
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # توليد الأسئلة
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_return_sequences=min(2, num_questions),
                    no_repeat_ngram_size=2,
                    num_beams=4,
                    early_stopping=True
                )

            # فك ترميز المخرجات
            for output in outputs:
                decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                if len(all_questions) < num_questions:
                    # تنسيق السؤال حسب النوع
                    if question_type == 'multiple_choice':
                        formatted_question = f"السؤال: {decoded_output}\nأ) خيار 1\nب) خيار 2\nج) خيار 3\nد) خيار 4\nالإجابة الصحيحة: أ"
                    elif question_type == 'essay':
                        formatted_question = f"السؤال: {decoded_output}\nإرشادات للإجابة: اكتب إجابة شاملة ومفصلة."
                    else:  # flashcards
                        formatted_question = f"السؤال: {decoded_output}\nالإجابة: إجابة السؤال"
                    
                    all_questions.append(formatted_question)

        return "\n\n".join(all_questions) if all_questions else None
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

@app.route('/auth/verify', methods=['POST'])
def verify_token():
    """التحقق من صحة رمز المصادقة."""
    try:
        if not firebase_enabled:
            return jsonify({'error': 'المصادقة معطلة حالياً'}), 503

        id_token = request.json.get('token')
        if not id_token:
            return jsonify({'error': 'لم يتم توفير رمز المصادقة'}), 400

        decoded_token = verify_firebase_token(id_token)
        if not decoded_token:
            return jsonify({'error': 'رمز المصادقة غير صالح'}), 401

        return jsonify({
            'user_id': decoded_token['uid'],
            'email': decoded_token.get('email', ''),
            'name': decoded_token.get('name', '')
        })
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        return jsonify({'error': 'حدث خطأ في التحقق من الرمز'}), 500

@app.route('/questions/history', methods=['GET'])
def get_questions_history():
    """استرجاع سجل الأسئلة للمستخدم."""
    try:
        if not firebase_enabled:
            return jsonify({'error': 'هذه الميزة غير متوفرة حالياً'}), 503

        id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not id_token:
            return jsonify({'error': 'يجب تسجيل الدخول'}), 401

        decoded_token = verify_firebase_token(id_token)
        if not decoded_token:
            return jsonify({'error': 'جلسة غير صالحة'}), 401

        user_id = decoded_token['uid']
        questions_ref = db.collection('questions')
        docs = questions_ref.where('user_id', '==', user_id).order_by(
            'timestamp', direction=firestore.Query.DESCENDING
        ).limit(10).get()

        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                'id': doc.id,
                'filename': data.get('filename'),
                'timestamp': data.get('timestamp'),
                'questions': data.get('questions')
            })

        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({'error': 'حدث خطأ أثناء تحميل السجل'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """معالجة رفع الملف وتوليد الأسئلة."""
    try:
        # التحقق من المصادقة
        if firebase_enabled:
            id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
            if not id_token:
                return jsonify({'error': 'يجب تسجيل الدخول'}), 401

            decoded_token = verify_firebase_token(id_token)
            if not decoded_token:
                return jsonify({'error': 'جلسة غير صالحة'}), 401

            user_id = decoded_token['uid']
        else:
            user_id = None

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
        questions = generate_questions_with_t5(text, num_questions, question_type)
        if not questions:
            logger.error("Failed to generate questions")
            return jsonify({'error': 'فشل في توليد الأسئلة'}), 500

        # حفظ في Firebase إذا كان المستخدم مسجل الدخول
        document_id = None
        if user_id:
            try:
                document_id = save_to_firebase(user_id, questions, file.filename)
                logger.info(f"Questions saved to Firebase with ID: {document_id}")
            except Exception as e:
                logger.error(f"Failed to save to Firebase: {str(e)}")

        # حذف الملف المؤقت
        try:
            os.remove(filepath)
            logger.info(f"Temporary file removed: {filepath}")
        except Exception as e:
            logger.error(f"Failed to remove temporary file: {str(e)}")

        return jsonify({
            'success': True,
            'questions': questions,
            'document_id': document_id
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
