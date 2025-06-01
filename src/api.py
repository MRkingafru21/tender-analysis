from fastapi import FastAPI
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from catboost import CatBoostClassifier
import joblib
import os
import json
import pandas as pd
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Загрузка моделей
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')

print("Загрузка CatBoost модели...")
model = CatBoostClassifier()
model.load_model(os.path.join(models_dir, 'catboost_model.cbm'))
print("CatBoost модель загружена!")

# Загрузка BERT
bert_dir = os.path.join(models_dir, 'bert_model')
print(f"Загрузка BERT модели из {bert_dir}...")

# Загрузка конфигурации BERT
bert_config = BertConfig.from_pretrained(bert_dir)

# Создаем модель с правильным размером словаря
bert_model = BertModel(bert_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

# Загружаем веса модели
weights_path = os.path.join(bert_dir, 'pytorch_model.bin')
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

# Фильтруем ключи, соответствующие текущей модели
filtered_state_dict = {k: v for k, v in state_dict.items() if k in bert_model.state_dict()}
bert_model.load_state_dict(filtered_state_dict, strict=False)
bert_model.eval()
print("BERT модель загружена!")

# Загрузка токенизатора
print("Загрузка токенизатора BERT...")
tokenizer = BertTokenizer.from_pretrained(bert_dir)
print("Токенизатор загружен!")

# Загрузка LabelEncoder
print("Загрузка LabelEncoder...")
le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
print("LabelEncoder загружен!")

# Загрузка конфигурации модели
print("Загрузка конфигурации...")
config_path = os.path.join(models_dir, 'config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    print("Конфигурация загружена!")
else:
    model_config = {"optimal_threshold": 0.5}
    print("Конфигурация не найдена, используется значение по умолчанию")

# Определяем Pydantic модель для входных данных
class TenderRequest(BaseModel):
    tender_name: str
    start_price: float
    tender_security: float
    advance_money: float
    procedure: str
    legislation: str
    currency: str
    publication_date: str

def get_bert_embeddings(text: str):
    """Генерация BERT-эмбеддингов для одного текста"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Mean-pooling
    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    
    return embeddings

def prepare_features(request: TenderRequest):
    """Подготовка признаков для модели"""
    # Создаем DataFrame из запроса
    data = {
        'tender_name': [request.tender_name],
        'start_price': [request.start_price],
        'tender_security': [request.tender_security],
        'advance_money': [request.advance_money],
        'procedure': [request.procedure],
        'legislation': [request.legislation],
        'currency': [request.currency],
        'publication_date': [request.publication_date]
    }
    
    df = pd.DataFrame(data)
    
    # Преобразования как в обучении
    df['tender_name_len'] = df['tender_name'].apply(len)
    df['word_count'] = df['tender_name'].apply(lambda x: len(x.split()))
    
    # Обработка процентов
    if isinstance(request.advance_money, str) and '%' in request.advance_money:
        df['advance_money'] = float(request.advance_money.replace('%', '').strip()) / 100
    else:
        df['advance_money'] = float(request.advance_money)
    
    # Расчет отношений
    security = request.tender_security
    if security == 0:
        security = 1e-9  # защита от деления на ноль
    
    df['security_percent'] = security / request.start_price
    df['price_security_ratio'] = request.start_price / security
    
    # Преобразование даты
    df['publication_date'] = pd.to_datetime(request.publication_date)
    df['publication_year'] = df['publication_date'].dt.year
    df['publication_month'] = df['publication_date'].dt.month
    df['publication_day'] = df['publication_date'].dt.day
    df['publication_dayofweek'] = df['publication_date'].dt.dayofweek
    
    # Кодирование категориальных признаков
    df['procurement_type'] = le.transform([request.procedure])[0]
    df['legislation_type'] = le.transform([request.legislation])[0]
    df['currency_type'] = le.transform([request.currency])[0]
    
    # Бинарные признаки
    df['has_advance'] = 1 if df['advance_money'].iloc[0] > 0 else 0
    df['has_security'] = 1 if security > 0 else 0
    df['has_winner'] = 0  # По умолчанию
    
    # Генерация BERT-эмбеддингов
    text_embeddings = get_bert_embeddings(request.tender_name)
    
    # Подготовка числовых признаков
    numeric_features = [
        'start_price', 'tender_security', 'advance_money', 'procurement_type',
        'legislation_type', 'currency_type', 'tender_name_len', 'word_count',
        'security_percent', 'price_security_ratio', 'publication_year',
        'publication_month', 'publication_day', 'publication_dayofweek',
        'has_advance', 'has_security', 'has_winner'
    ]
    
    numeric = df[numeric_features].values
    features = np.concatenate([text_embeddings, numeric], axis=1)
    
    return features

# Ключевой эндпоинт для предсказаний
@app.post("/predict")
async def predict_tender(request: TenderRequest):
    try:
        print(f"Получен запрос: {request}")
        features = prepare_features(request)
        proba = model.predict_proba(features)[0][1]
        
        # Оптимальный порог
        threshold = model_config.get("optimal_threshold", 0.5)
        prediction = 1 if proba >= threshold else 0
        confidence = proba if prediction == 1 else 1 - proba
        
        return {
            "prediction": int(prediction),
            "probability": float(proba),
            "confidence": float(confidence),
            "threshold": float(threshold)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Эндпоинт для проверки работоспособности
@app.get("/health")
def health_check():
    return {"status": "OK", "timestamp": datetime.now().isoformat()}