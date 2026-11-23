from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

from app.functions.classification import preprocess, predict,classify_booking

router = APIRouter()


class Request(BaseModel):
    booking: dict[str, Any]
    
@router.get('/')
async def home():
    return {
        'message':'Hello World !'
    }
@router.post("/classify")
def predict_route(request: Request):
    return classify_booking(request.booking)
