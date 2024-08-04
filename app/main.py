from flask import Flask, render_template, request
from utils import clean_text_data, postprocess_response 
from model import MedicalChatbotModel
from preprocess_data import get_cleaned_data

import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model_path = 'model/finetuned_model'
df_questions = get_cleaned_data()

chatbot_model = MedicalChatbotModel(model_name=model_path, df=df_questions)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/get')
def get_model_response():
    user_query = request.args.get('query')
    preprocessed_query = clean_text_data(user_query)
    predictions = chatbot_model.predict_category(preprocessed_query)
    response = postprocess_response(predictions)
    return response


if __name__=="__main__":
    app.run(debug=True)