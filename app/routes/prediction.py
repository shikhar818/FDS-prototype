from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any
from pathlib import Path
from io import BytesIO
import pandas as pd
import uuid
from app.functions.pipeline import run_prediction_pipeline


router = APIRouter(tags=["Prediction"])

class BookingRequest(BaseModel):
    booking: dict[str, Any]

@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile=File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported."
            )
        contents = await file.read()
        booking_df = pd.read_csv(
            BytesIO(contents)
        )
        if booking_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV contains no records.")
        result_df = run_prediction_pipeline(
            booking_df)
        output_dir=Path("outputs")
        output_dir.mkdir(
            parents=True,
            exist_ok=True)
        filename=f"predictions_{uuid.uuid4().hex[:8]}.csv"
        output_path = output_dir / filename
        result_df.to_csv(
            output_path,
            index=False
        )
        return FileResponse(
            path=output_path,
            media_type="text/csv",
            filename=filename
        )
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV is empty or invalid."
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )