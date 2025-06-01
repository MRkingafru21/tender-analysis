import sys
import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, log_loss
)
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from catboost import CatBoostClassifier, Pool, cv
from datetime import datetime
import json
import re

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт модуля
from src.data_preprocessing import preprocess_data

# Конфигурация
CONFIG = {
    "bert_model": "DeepPavlov/rubert-base-cased",
    "batch_size": 16,
    "max_length": 128,
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "catboost_iterations": 2000,
    "learning_rate": 0.05,
    "early_stopping_rounds": 100,
    "eval_metric": "F1",
    "output_dir": "../output",
    "model_dir": "../models",
    "report_dir": "../reports",
    "hyperparam_tuning": True
}

# Проверяем доступность GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Создание директорий
for dir_path in [CONFIG["output_dir"], CONFIG["model_dir"], CONFIG["report_dir"]]:
    os.makedirs(dir_path, exist_ok=True)

# Инициализация BERT
tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model"])
bert_model = BertModel.from_pretrained(CONFIG["bert_model"]).to(device)

def get_bert_embeddings(texts, batch_size=CONFIG["batch_size"]):
    """Генерация BERT-эмбеддингов с батчингом"""
    embeddings = []
    bert_model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=CONFIG["max_length"],
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
        batch_embeddings = sum_embeddings / sum_mask
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Оценка модели на тестовом наборе с кастомным порогом"""
    # Предсказания
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Расчет метрик
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "threshold": threshold
    }
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_proba

def find_optimal_threshold(model, X_val, y_val):
    """Поиск оптимального порога классификации по F1"""
    y_val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    
    for thresh in thresholds:
        preds = (y_val_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_val, preds))
    
    best_thresh = thresholds[np.argmax(f1_scores)]
    return best_thresh, max(f1_scores)

def plot_learning_curve(history, report_path):
    """Визуализация кривой обучения"""
    if not history:
        print("No learning history available")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Получаем метрики из истории
    train_metric = history['learn'][CONFIG['eval_metric']]
    val_metric = history['validation'][CONFIG['eval_metric']]
    
    # Строим график
    plt.plot(train_metric, label='Train')
    plt.plot(val_metric, label='Validation')
    
    # Находим точку ранней остановки
    best_iter = np.argmin(val_metric) if CONFIG['eval_metric'] == 'Logloss' else np.argmax(val_metric)
    plt.axvline(x=best_iter, color='r', linestyle='--', 
                label=f'Best iteration: {best_iter}')
    
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel(CONFIG['eval_metric'])
    plt.legend()
    plt.grid(True)
    
    # Сохраняем график
    plt.savefig(os.path.join(report_path, 'learning_curve.png'))
    plt.close()

def plot_metrics(metrics, cm, report_path):
    """Визуализация метрик и сохранение графиков"""
    # График матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Не успешен', 'Успешен'],
                yticklabels=['Не успешен', 'Успешен'])
    plt.xlabel('Предсказание')
    plt.ylabel('Истина')
    plt.title('Матрица ошибок')
    plt.savefig(os.path.join(report_path, 'confusion_matrix.png'))
    plt.close()
    
    # График метрик
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {k: v for k, v in metrics.items() if k not in ['threshold', 'log_loss']}
    sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()))
    plt.title('Метрики классификации')
    plt.ylabel('Значение')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(report_path, 'metrics.png'))
    plt.close()

def save_report(metrics, cm, feature_importance, config, report_path):
    """Сохранение отчета о модели"""
    # Сохранение метрик
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance.tolist()
    }
    
    with open(os.path.join(report_path, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Текстовый отчет
    with open(os.path.join(report_path, 'summary.txt'), 'w') as f:
        f.write("Отчет о модели классификации тендеров\n")
        f.write("====================================\n\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nКонфигурация:\n")
        for key, value in config.items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\nМетрики:\n")
        for metric, value in metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        
        f.write("\nМатрица ошибок:\n")
        f.write(np.array2string(cm))
        
        f.write("\n\nТоп-10 важных признаков:\n")
        sorted_indices = np.argsort(feature_importance)[::-1][:10]
        for i in sorted_indices:
            f.write(f"- Признак {i}: {feature_importance[i]:.4f}\n")

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Оптимизация гиперпараметров CatBoost"""
    print("Начало оптимизации гиперпараметров...")
    
    # Параметры для перебора
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'iterations': [500, 1000, 1500]
    }
    
    best_score = 0
    best_params = {}
    
    for depth in param_grid['depth']:
        for lr in param_grid['learning_rate']:
            for reg in param_grid['l2_leaf_reg']:
                for iters in param_grid['iterations']:
                    model = CatBoostClassifier(
                        iterations=iters,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=reg,
                        eval_metric='F1',
                        early_stopping_rounds=100,
                        random_seed=CONFIG["random_state"],
                        verbose=False
                    )
                    
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        use_best_model=True
                    )
                    
                    # Оценка на валидации
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'iterations': iters,
                            'learning_rate': lr,
                            'depth': depth,
                            'l2_leaf_reg': reg
                        }
                        print(f"Новый лучший F1: {best_score:.4f} с параметрами: {best_params}")
    
    print(f"Лучшие параметры: {best_params}, F1: {best_score:.4f}")
    return best_params

def main():
    # Начало времени выполнения
    start_time = datetime.now()
    
    # Получаем абсолютный путь к данным
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'tender_data.csv')
    
    # Загрузка и предобработка данных
    print("Загрузка и предобработка данных...")
    df, le = preprocess_data(data_path)
    
    # Анализ дисбаланса классов
    class_counts = df['is_successful'].value_counts()
    print(f"Распределение классов:\n{class_counts}")
    
    # Расчет весов классов
    class_weights = {
        0: len(df) / (2 * class_counts[0]),
        1: len(df) / (2 * class_counts[1])
    }
    print(f"Веса классов: {class_weights}")
    
    # Разделение данных
    X = df.drop('is_successful', axis=1)
    y = df['is_successful']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"],
        stratify=y
    )
    
    # Разделение train на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=CONFIG["val_size"],
        random_state=CONFIG["random_state"],
        stratify=y_train
    )
    
    # Генерация BERT-эмбеддингов
    print("Генерация BERT-эмбеддингов для тренировочных данных...")
    train_text_embeddings = get_bert_embeddings(X_train['tender_name'])
    
    print("Генерация BERT-эмбеддингов для валидационных данных...")
    val_text_embeddings = get_bert_embeddings(X_val['tender_name'])
    
    print("Генерация BERT-эмбеддингов для тестовых данных...")
    test_text_embeddings = get_bert_embeddings(X_test['tender_name'])
    
    # Подготовка числовых признаков
    numeric_features = [
        'start_price', 'tender_security', 'advance_money', 'procurement_type',
        'legislation_type', 'currency_type', 'tender_name_len', 'word_count',
        'security_percent', 'price_security_ratio', 'publication_year',
        'publication_month', 'publication_day', 'publication_dayofweek',
        'has_advance', 'has_security', 'has_winner'
    ]
    
    # Сборка финальных наборов данных
    def prepare_features(text_emb, df, numeric_cols):
        numeric = df[numeric_cols].values
        return np.concatenate([text_emb, numeric], axis=1)
    
    X_train_final = prepare_features(train_text_embeddings, X_train, numeric_features)
    X_val_final = prepare_features(val_text_embeddings, X_val, numeric_features)
    X_test_final = prepare_features(test_text_embeddings, X_test, numeric_features)
    
    # Оптимизация гиперпараметров
    if CONFIG["hyperparam_tuning"]:
        best_params = tune_hyperparameters(
            X_train_final, y_train, 
            X_val_final, y_val
        )
        # Обновляем конфигурацию
        CONFIG.update({
            "catboost_iterations": best_params['iterations'],
            "learning_rate": best_params['learning_rate'],
            "depth": best_params['depth'],
            "l2_leaf_reg": best_params['l2_leaf_reg']
        })
    
    # Конфигурация GPU для CatBoost
    gpu_params = {
        'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
        'devices': '0',
        'border_count': 128,
    }
    
    # Обучение CatBoost с валидацией и ранней остановкой
    print("Обучение CatBoost модели...")
    model = CatBoostClassifier(
        iterations=CONFIG["catboost_iterations"],
        learning_rate=CONFIG["learning_rate"],
        depth=CONFIG.get("depth", 6),
        l2_leaf_reg=CONFIG.get("l2_leaf_reg", 3),
        eval_metric=CONFIG["eval_metric"],
        early_stopping_rounds=CONFIG["early_stopping_rounds"],
        verbose=100,
        random_seed=CONFIG["random_state"],
        class_weights=class_weights,
        **gpu_params
    )
    
    model.fit(
        X_train_final, y_train,
        eval_set=(X_val_final, y_val),
        use_best_model=True
    )
    
    # Получаем историю обучения для визуализации
    history = model.get_evals_result()
    
    # Поиск оптимального порога классификации
    best_threshold, best_f1 = find_optimal_threshold(model, X_val_final, y_val)
    print(f"Оптимальный порог: {best_threshold:.4f}, F1: {best_f1:.4f}")
    
    # Оценка модели на тестовых данных
    print("Оценка модели на тестовых данных...")
    test_metrics, test_cm, test_proba = evaluate_model(
        model, X_test_final, y_test, threshold=best_threshold
    )
    
    # Важность признаков
    feature_importance = model.get_feature_importance()
    
    # Создание отчета
    report_path = os.path.join(
        CONFIG["report_dir"], 
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(report_path, exist_ok=True)
    
    print("Создание отчетов...")
    # Визуализация кривой обучения
    if history:
        plot_learning_curve(history, report_path)
    
    plot_metrics(test_metrics, test_cm, report_path)
    save_report(test_metrics, test_cm, feature_importance, CONFIG, report_path)
    
    # Сохранение модели и артефактов
    print("Сохранение моделей и артефактов...")
    model.save_model(os.path.join(CONFIG["model_dir"], 'catboost_model.cbm'))
    joblib.dump(le, os.path.join(CONFIG["model_dir"], 'label_encoder.pkl'))
    
    # Сохранение BERT
    bert_dir = os.path.join(CONFIG["model_dir"], 'bert_model')
    os.makedirs(bert_dir, exist_ok=True)
    torch.save(bert_model.state_dict(), os.path.join(bert_dir, 'pytorch_model.bin'))
    tokenizer.save_pretrained(bert_dir)
    
    # Сохранение конфигурации
    with open(os.path.join(CONFIG["model_dir"], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Время выполнения
    duration = datetime.now() - start_time
    print(f"Процесс завершен! Общее время: {duration}")
    print(f"Метрики на тестовых данных: {test_metrics}")
    print(f"Отчеты сохранены в: {report_path}")

if __name__ == "__main__":
    main()