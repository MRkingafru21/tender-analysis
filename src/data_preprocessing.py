import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

def preprocess_data(filepath):
    # Загрузка данных
    df = pd.read_csv(filepath)
    
    # Создание целевой переменной
    df['is_successful'] = df['selection_phase'].apply(
        lambda x: 0 if "несостоявшейся" in str(x) else 1
    )
    
    # Обработка advance_money
    df['advance_money'] = df['advance_money'].apply(
        lambda x: float(str(x).replace('%', '').strip()) / 100 
        if '%' in str(x) else float(x)
    )
    
    # Обработка final_price
    df['final_price'] = df['final_price'].apply(
        lambda x: float(re.sub(r'[^\d.]', '', str(x))) if str(x) != 'nan' else np.nan
    )
    
    # Новые признаки
    df['tender_name_len'] = df['tender_name'].apply(lambda x: len(str(x)))
    df['word_count'] = df['tender_name'].apply(lambda x: len(str(x).split()))
    df['security_percent'] = df['tender_security'] / df['start_price']
    df['price_security_ratio'] = df['start_price'] / df['tender_security']
    
    # Временные признаки
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    df['publication_year'] = df['publication_date'].dt.year
    df['publication_month'] = df['publication_date'].dt.month
    df['publication_day'] = df['publication_date'].dt.day
    df['publication_dayofweek'] = df['publication_date'].dt.dayofweek
    
    # Кодирование категориальных признаков
    le = LabelEncoder()
    df['procurement_type'] = le.fit_transform(df['procedure'])
    df['legislation_type'] = le.fit_transform(df['legislation'])
    df['currency_type'] = le.fit_transform(df['currency'])
    
    # Бинарные признаки
    df['has_advance'] = df['advance_money'].apply(lambda x: 1 if x > 0 else 0)
    df['has_security'] = df['tender_security'].apply(lambda x: 1 if x > 0 else 0)
    
    # Признаки на основе победителя
    df['has_winner'] = df['winner_name'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # Выбор признаков
    features = [
        'tender_name', 'start_price', 'tender_security', 'advance_money',
        'procurement_type', 'legislation_type', 'currency_type',
        'tender_name_len', 'word_count', 'security_percent', 'price_security_ratio',
        'publication_year', 'publication_month', 'publication_day', 'publication_dayofweek',
        'has_advance', 'has_security', 'has_winner'
    ]
    target = 'is_successful'
    
    return df[features + [target]], le