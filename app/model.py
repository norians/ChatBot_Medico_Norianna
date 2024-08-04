from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

class MedicalChatbotModel:
    def __init__(self, model_name: str, df: pd.DataFrame = None):
        self.model = SentenceTransformer(model_name)
        if df is not None:
            self.responses = df['answer'].tolist()  
            self.responses_embeddings = self.model.encode(self.responses, convert_to_tensor=True)
        else:
            self.responses = [] 
            self.responses_embeddings = None

    def fine_tune(self, df: pd.DataFrame, num_epochs: int = 1):
        # Convertir los ejemplos en InputExamples
        answers = df['answer'].tolist()
        train_examples = []
        for _, row in df.iterrows():
            if pd.notna(row['message']) and pd.notna(row['answer']):
                negative_answer = random.choice([ans for ans in answers if ans != row['answer']])
                train_examples.append(InputExample(texts=[row['message'], row['answer'], negative_answer]))
        # Crear el dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model=self.model)
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        # Entrenar el modelo
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps) 
        self.model.save('model/finetuned_model')

    def predict_category(self, question: str) -> str:
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(question_embedding, self.responses_embeddings)[0]
        best_match_idx = np.argmax(scores)
        return self.responses[best_match_idx]
