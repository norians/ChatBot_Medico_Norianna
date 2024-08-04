import nltk
import re

def clean_text_data(text):
    text = text.lower()

    # Eliminar caracteres especiales, números, and signos de puntuación
    text = re.sub(r'[^A-Za-z\s]', '', text)

    #Eliminar urls
    text = re.sub(r'http\S+', '', text)

    #Quitar espacios en blanco
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def postprocess_response(response: str) -> str:
    if response != "":       
        # Añadir puntuación y formatear adecuadamente
        response = response.capitalize() + "."
    else:
        response = "Sorry, I don't know how to answer your question."
    return response