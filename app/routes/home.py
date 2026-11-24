from fastapi import APIRouter 


router = APIRouter(tags=["Home"])



@router.get('/')
async def home():
    return {
        'success':True,
        'message':'Hotel Revenue and Cancellation Prediction API',
        'version':'1.0.0-alpha',
        "endpoints":{
            "documentation":"/docs",
            "batch_prediction":"/predict/batch"
        }
    }