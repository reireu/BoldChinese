from fastapi import APIRouter, File, Form, UploadFile
from app.services.kaldi_service import analyze_pronunciation
import shutil
import tempfile

router = APIRouter()

@router.post("/analyze")
async def analyze(audio: UploadFile = File(...), text: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        audio_path = tmp.name

    result = analyze_pronunciation(audio_path, text)
    return result
