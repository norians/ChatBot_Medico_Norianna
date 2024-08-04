import pandas as pd
from model import MedicalChatbotModel
from preprocess_data import get_cleaned_data

# Cargar datos de entrenamiento
df_questions = get_cleaned_data()

# Crear el modelo
model_name = 'all-MiniLM-L6-v2'

model = MedicalChatbotModel(model_name=model_name, df=df_questions)

# Fine-tune
model.fine_tune(df=df_questions)