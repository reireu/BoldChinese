import asyncio
import aiohttp
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import magic
from dataclasses import dataclass
import logging
from fastapi import status
from app.routers.analysis import AnalysisDetails, AnalysisResult, AudioAnalysisException, AudioValidationException, KaldiServiceException
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

# 定数定義
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg"]
MAX_FILE_SIZE_MB = 50
TIMEOUT_DURATION = 120
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0
MIME_MAGIC = magic.Magic(mime=True)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalysisException(Exception):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.original_error is not None:
            return f'AudioAnalysisException: {self.message} ({self.original_error})'
        return f'AudioAnalysisException: {self.message}'

class AudioValidationException(AudioAnalysisException):
    def __init__(self, message: str):
        super().__init__(message)

class KaldiServiceException(AudioAnalysisException):
    def __init__(self, message: str, status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        self.status_code = status_code
        super().__init__(message, original_error)
    
    def __str__(self) -> str:
        status_info = f' (Status: {self.status_code})' if self.status_code is not None else ''
        return f'KaldiServiceException: {self.message}{status_info}'

@dataclass
class AnalysisDetails:
    pronunciation: float
    intonation: float
    rhythm: float
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'AnalysisDetails':
        try:
            return cls(
                pronunciation=float(json_data['pronunciation']),
                intonation=float(json_data['intonation']),
                rhythm=float(json_data['rhythm'])
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f'Invalid JSON format for AnalysisDetails: {e}')
    
    def to_json(self) -> Dict[str, float]:
        return {
            'pronunciation': self.pronunciation,
            'intonation': self.intonation,
            'rhythm': self.rhythm
        }
    
    def is_valid(self) -> bool:
        return (0 <= self.pronunciation <= 100 and
                0 <= self.intonation <= 100 and
                0 <= self.rhythm <= 100)
    
    def __str__(self) -> str:
        return f'AnalysisDetails(pronunciation: {self.pronunciation}, intonation: {self.intonation}, rhythm: {self.rhythm})'

@dataclass
class AnalysisResult:
    success: bool
    score: Optional[float] = None
    feedback: Optional[str] = None
    details: Optional[AnalysisDetails] = None
    error_message: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'AnalysisResult':
        details = None
        if json_data.get('details') is not None:
            details = AnalysisDetails.from_json(json_data['details'])
        
        return cls(
            success=bool(json_data['success']),
            score=float(json_data['score']) if json_data.get('score') is not None else None,
            feedback=json_data.get('feedback'),
            details=details,
            error_message=json_data.get('error')
        )
    
    @classmethod
    def error(cls, message: str, original_error: Optional[Exception] = None) -> 'AnalysisResult':
        error_msg = message + (f' ({original_error})' if original_error is not None else '')
        return cls(
            success=False,
            error_message=error_msg
        )

router = APIRouter()

@router.post("/analyze-audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    service = AnalyzeService()
    try:
        # 一時ファイルの作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.seek(0)
            
            # 分析の実行
            result = await service.analyze_audio(temp_file.name)
            
            if not result.success:
                raise HTTPException(status_code=400, detail=result.error_message)
                
            return result
            
    except AudioValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KaldiServiceException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        service.dispose()
        
class AnalyzeService:
    def __init__(self):
        self.KALDI_SERVICE_URL = os.environ.get('KALDI_SERVICE_URL', 'http://localhost:8080/analyze')
        self._cache: Dict[str, AnalysisResult] = {}
    
    async def analyze_audio_with_retry(self, audio_file: Union[str, Path]) -> AnalysisResult:
        for i in range(RETRY_ATTEMPTS):
            try:
                return await self.analyze_audio(audio_file)
            except KaldiServiceException as e:
                if i == RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(RETRY_DELAY * (i + 1))
        raise KaldiServiceException('Max retries exceeded')

    async def _validate_audio_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise AudioValidationException(f'Audio file does not exist: {file_path}')
        
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise AudioValidationException(
                f'Audio file size exceeds the limit of {MAX_FILE_SIZE_MB}MB.'
            )
        
        try:
            mime_type = MIME_MAGIC.from_file(str(file_path))
            if mime_type not in ALLOWED_AUDIO_TYPES:
                raise AudioValidationException(f"不正なファイル形式です: {mime_type}")
        except Exception as e:
            raise AudioValidationException(f"ファイル形式の検証に失敗しました: {str(e)}")

    async def analyze_audio(self, audio_file: Union[str, Path]) -> AnalysisResult:
        audio_path = Path(audio_file)
        logger.info(f'Starting audio analysis for: {audio_path}')
        
        try:
            await self._validate_audio_file(audio_path)
            raw_response = await self._send_to_kaldi_service(audio_path)
            return self._process_analysis_result(raw_response)
            
        except AudioValidationException as e:
            logger.error(f'Audio validation error: {e.message}')
            return AnalysisResult.error(f'Invalid audio file: {e.message}', e)
            
        except Exception as e:
            logger.error(f'Analysis error: {str(e)}')
            return AnalysisResult.error(str(e), e)

    async def _send_to_kaldi_service(self, audio_file: Path) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT_DURATION)) as session:
                with open(audio_file, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('audio_file', f, filename=audio_file.name)
                    
                    async with session.post(self.KALDI_SERVICE_URL, data=data) as response:
                        if response.status != 200:
                            raise KaldiServiceException(
                                'Kaldi service error',
                                status_code=response.status
                            )
                        return await response.json()
                        
        except Exception as e:
            raise KaldiServiceException(f'Service communication error: {str(e)}', original_error=e)

    def _process_analysis_result(self, raw_response: Dict[str, Any]) -> AnalysisResult:
        try:
            return AnalysisResult.from_json(raw_response)
        except Exception as e:
            return AnalysisResult.error(f'Error processing result: {str(e)}', e)

    def dispose(self):
        self._cache.clear()

async def main():
    service = AnalyzeService()
    try:
        result = await service.analyze_audio("test.wav")
        print(result)
    finally:
        service.dispose()

if __name__ == "__main__":
    asyncio.run(main())