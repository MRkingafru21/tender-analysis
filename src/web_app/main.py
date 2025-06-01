import os
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import json

app = FastAPI()

# Определяем абсолютные пути
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

print(f"Current directory: {current_dir}")
print(f"Static directory: {static_dir}")
print(f"Templates directory: {templates_dir}")

# Создаем директории, если их нет
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Монтируем статику и шаблоны
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# URL API
API_URL = "http://localhost:8000/predict"

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def web_predict(
    request: Request,
    tender_name: str = Form(...),
    start_price: float = Form(...),
    tender_security: float = Form(...),
    advance_money: str = Form(...),
    procedure: str = Form(...),
    legislation: str = Form("44-ФЗ"),
    currency: str = Form("RUB"),
    publication_date: str = Form(...)
):
    # Формируем данные для API
    data = {
        "tender_name": tender_name,
        "start_price": start_price,
        "tender_security": tender_security,
        "advance_money": advance_money,
        "procedure": procedure,
        "legislation": legislation,
        "currency": currency,
        "publication_date": publication_date
    }
    
    try:
        # Отправляем запрос к API
        response = requests.post(API_URL, json=data)
        
        # Проверяем статус ответа
        if response.status_code != 200:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": f"Ошибка API: {response.status_code}",
                    "form_data": data
                }
            )
        
        result = response.json()
        
        # Если API вернуло ошибку
        if "error" in result:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": result["error"],
                    "form_data": data
                }
            )
        
        # Проверяем наличие всех необходимых полей
        required_fields = ["prediction", "probability", "confidence", "threshold"]
        if not all(field in result for field in required_fields):
            missing = [field for field in required_fields if field not in result]
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": f"В ответе API отсутствуют поля: {', '.join(missing)}",
                    "form_data": data
                }
            )
        
        # Форматируем результат
        result["probability_percent"] = round(result["probability"] * 100, 2)
        result["confidence_percent"] = round(result["confidence"] * 100, 2)
        result["success_text"] = "Успешен" if result["prediction"] else "Не успешен"
        result["success_class"] = "success" if result["prediction"] else "failure"
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "form_data": data
            }
        )
            
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Ошибка связи с API: {str(e)}",
                "form_data": data
            }
        )