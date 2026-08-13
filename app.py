from pathlib import Path
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from textSummarizer.pipeline.prediction import PredictionPipeline


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI Text Summarizer",
    description="Text summarization application using FastAPI.",
    version="1.0.0",
)


@app.get("/")
async def home():
    index_file = BASE_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found.",
        )

    return FileResponse(index_file)


@app.get("/style.css")
async def stylesheet():
    css_file = BASE_DIR / "style.css"

    if not css_file.exists():
        raise HTTPException(
            status_code=404,
            detail="style.css was not found.",
        )

    return FileResponse(
        css_file,
        media_type="text/css",
    )


@app.get("/train")
async def training():
    try:
        exit_code = os.system("python main.py")

        if exit_code != 0:
            return Response(
                content="Training failed. Check the terminal logs.",
                status_code=500,
            )

        return Response(
            content="Training completed successfully.",
            status_code=200,
        )

    except Exception as error:
        return Response(
            content=f"Training error: {error}",
            status_code=500,
        )


@app.get("/predict")
async def predict_route(text: str):
    cleaned_text = text.strip()

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Please provide text to summarize.",
        )

    try:
        prediction_pipeline = PredictionPipeline()
        summary = prediction_pipeline.predict(cleaned_text)

        if isinstance(summary, list):
            summary = summary[0] if summary else ""

        if isinstance(summary, dict):
            if "summary_text" in summary:
                summary = summary["summary_text"]
            elif "summary" in summary:
                summary = summary["summary"]

        return {
            "summary": summary,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
