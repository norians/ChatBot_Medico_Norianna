import xml.etree.ElementTree as ET
import pandas as pd
from utils import clean_text_data

def get_cleaned_data():
    data_train_xml_1 = ET.parse('data/TREC-2017-LiveQA-Medical-Train-1.xml')
    root_1 = data_train_xml_1.getroot()
    data_train_xml_2 = ET.parse('data/TREC-2017-LiveQA-Medical-Train-2.xml')
    root_2 = data_train_xml_2.getroot()

    data_train = []

    for registro in root_1.findall('NLM-QUESTION'):
        message = registro.find('MESSAGE').text 
        sub_question = registro.find('SUB-QUESTIONS').find('SUB-QUESTION')
        type_element = sub_question.find('ANNOTATIONS').find('TYPE').text 
        answer = sub_question.find('ANSWERS').find('ANSWER').text
        data_train.append({
            'message': message,
            'type': type_element,
            'answer': answer
        })

    for registro in root_2.findall('NLM-QUESTION'):
        message = registro.find('MESSAGE').text 
        sub_question = registro.find('SUB-QUESTIONS').find('SUB-QUESTION')
        type_element = sub_question.find('ANNOTATIONS').find('TYPE').text 
        answer = sub_question.find('ANSWERS').find('ANSWER').text
        data_train.append({
            'message': message,
            'type': type_element,
            'answer': answer
        })

    df_questions = pd.DataFrame(data_train)

    df_questions = df_questions.dropna(subset=['message'])

    df_questions['message'] = df_questions['message'].apply(clean_text_data)
    df_questions['answer'] = df_questions['answer'].apply(clean_text_data)

    return df_questions
