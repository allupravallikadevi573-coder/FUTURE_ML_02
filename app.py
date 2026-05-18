from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  # This allows your index.html webpage to talk to this backend safely

# 1. Load your successful Kaggle-trained assets
model = joblib.load('it_ticket_classifier_model.pkl')
tfidf = joblib.load('it_tfidf_vectorizer.pkl')

# 2. Re-apply your exact text-cleaning function from Kaggle
def clean_ticket_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# 3. Create the API Route for the web dashboard
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('ticket_text', '')
    
    if not raw_text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400
        
    # Process text through your NLP pipeline
    cleaned = clean_ticket_text(raw_text)
    vectorized = tfidf.transform([cleaned])
    
    # Generate the prediction
    prediction = model.predict(vectorized)[0]
    
    # Return the automated business category mapping
    return jsonify({
        'status': 'success',
        'predicted_category': str(prediction)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)