import asyncio
import aiohttp
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import magic
from dataclasses import dataclass
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAnalysisException(Exception):
    """オーディオ分析サービス全般で発生する可能性のある例外の基底クラス"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.original_error is not None:
            return f'AudioAnalysisException: {self.message} ({self.original_error})'
        return f'AudioAnalysisException: {self.message}'


class AudioValidationException(AudioAnalysisException):
    """オーディオファイルのバリデーションエラーを示す例外"""
    
    def __init__(self, message: str):
        super().__init__(message)


class KaldiServiceException(AudioAnalysisException):
    """Kaldiサービスとの通信中に発生したエラーを示す例外"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        self.status_code = status_code
        super().__init__(message, original_error)
    
    def __str__(self) -> str:
        status_info = f' (Status: {self.status_code})' if self.status_code is not None else ''
        return f'KaldiServiceException: {self.message}{status_info}'


@dataclass
class AnalysisDetails:
    """分析の詳細情報を保持するクラス"""
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
    """分析結果を保持するクラス"""
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
        """エラー発生時に元のエラー情報を渡せるように改善"""
        error_msg = message + (f' ({original_error})' if original_error is not None else '')
        return cls(
            success=False,
            error_message=error_msg
        )
    
    def __str__(self) -> str:
        if self.success:
            return f'AnalysisResult(success: True, score: {self.score}, feedback: "{self.feedback}", details: {self.details})'
        else:
            return f'AnalysisResult(success: False, error_message: "{self.error_message}")'


class AnalyzeService:
    """オーディオ分析サービスのメインクラス"""
    
    # 定数をより具体的に
    KALDI_SERVICE_URL = os.environ.get('KALDI_SERVICE_URL', 'http://localhost:8080/analyze')
    MAX_FILE_SIZE_MB = 50  # デフォルト値として設定
    TIMEOUT_DURATION = 120  # 秒
    
    def __init__(self):
        # キャッシュの追加
        self._cache: Dict[str, AnalysisResult] = {}
    
    def _log(self, message: str, is_error: bool = False) -> None:
        """メッセージをログに出力します。"""
        if is_error:
            logger.error(f'AnalyzeService - {message}')
        else:
            logger.info(f'AnalyzeService - {message}')
    
    async def analyze_audio_with_retry(self, audio_file: Union[str, Path], 
                                     max_retries: int = 3, 
                                     retry_delay: float = 1.0) -> AnalysisResult:
        """リトライ機能付きでオーディオを分析"""
        for i in range(max_retries):
            try:
                return await self.analyze_audio(audio_file)
            except KaldiServiceException as e:
                if i == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (i + 1))
        
        raise KaldiServiceException('Max retries exceeded')
    
    async def preload_models(self) -> None:
        """モデルのプリロード（プレースホルダー）"""
        pass
    
    def dispose(self) -> None:
        """メモリ解放"""
        self._cache.clear()
    
    async def analyze_audio(self, audio_file: Union[str, Path]) -> AnalysisResult:
        """オーディオファイルを分析し、AnalysisResultを返します。"""
        audio_path = Path(audio_file)
        self._log(f'Starting audio analysis for: {audio_path}')
        
        try:
            # ファイルバリデーション
            await self._validate_audio_file(audio_path)
            
            # Kaldiサービスにリクエスト送信
            raw_response = await self._send_to_kaldi_service(audio_path)
            
            # 結果の解析と加工
            return self._process_analysis_result(raw_response)
            
        except AudioValidationException as e:
            self._log(f'Audio validation error: {e.message}', is_error=True)
            return AnalysisResult.error(f'Invalid audio file: {e.message}', original_error=e)
            
        except KaldiServiceException as e:
            self._log(f'Kaldi service communication error: {e.message}', is_error=True)
            return AnalysisResult.error(f'Failed to communicate with analysis service: {e.message}', original_error=e)
            
        except asyncio.TimeoutError as e:
            self._log(f'Request timed out: {e}', is_error=True)
            return AnalysisResult.error('Analysis request timed out. Please try again.', original_error=e)
            
        except Exception as e:
            self._log(f'An unexpected error occurred: {e}', is_error=True)
            return AnalysisResult.error(f'An unexpected error occurred during analysis: {e}', original_error=e)
    
    async def _validate_audio_file(self, file_path: Path) -> None:
        """オーディオファイルのバリデーションを行います。
        無効な場合は AudioValidationException をスローします。"""
        
        extension = file_path.suffix.lower()
        self._log(f'Validating file: {file_path}, extension: {extension}')
        
        if extension not in ['.wav', '.mp3']:
            raise AudioValidationException(
                f'Unsupported audio file format: {extension}. Only .wav and .mp3 are supported.'
            )
        
        # ファイルサイズのチェック
        if not file_path.exists():
            raise AudioValidationException(f'Audio file does not exist: {file_path}')
        
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise AudioValidationException(
                f'Audio file size exceeds the limit of {self.MAX_FILE_SIZE_MB}MB.'
            )
        
        mime_type = MIME_MAGIC.from_file(str(file_path))
        if mime_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不正なファイル形式です: {mime_type}"
            )
        
        self._log('ファイルのバリデーションが成功しました。')
    
    async def _send_to_kaldi_service(self, audio_file: Path) -> Dict[str, Any]:
        """Kaldiサービスにオーディオファイルを送信し、その生のレスポンスを返します。
        エラーが発生した場合は KaldiServiceException または TimeoutError をスローします。"""
        
        self._log(f'Sending audio file to Kaldi service: {audio_file}')
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.TIMEOUT_DURATION)) as session:
                with open(audio_file, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('audio_file', f, filename=audio_file.name)
                    
                    async with session.post(self.KALDI_SERVICE_URL, data=data) as response:
                        self._log(f'Received response from Kaldi service with status: {response.status}')
                        
                        if response.status != 200:
                            error_body = await response.text()
                            self._log(f'Kaldi service returned error: {error_body}', is_error=True)
                            raise KaldiServiceException(
                                f'Kaldi service returned an error. Status: {response.status}',
                                status_code=response.status,
                                original_error=error_body
                            )
                        
                        # レスポンスボディがJSON形式であることを期待
                        try:
                            return await response.json()
                        except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                            error_text = await response.text()
                            self._log(f'Failed to parse Kaldi service response as JSON: {error_text}', is_error=True)
                            raise KaldiServiceException(
                                'Failed to parse analysis result from service.',
                                original_error=e
                            )
        
        except aiohttp.ClientError as e:
            raise KaldiServiceException(f'Network error communicating with Kaldi service: {e}', original_error=e)
        
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(f'Request to Kaldi service timed out after {self.TIMEOUT_DURATION} seconds.')
        
        except Exception as e:
            raise KaldiServiceException(f'Failed to communicate with Kaldi service: {e}', original_error=e)
    
    def _process_analysis_result(self, raw_response: Dict[str, Any]) -> AnalysisResult:
        """Kaldiサービスからの生のレスポンスを解析し、AnalysisResult オブジェクトに変換します。"""
        
        self._log('Processing raw analysis result.')
        
        try:
            # Kaldiからの生のレスポンス構造に応じて、AnalysisResult.from_json に渡すデータを整形
            # 例: Kaldiが直接 'score', 'feedback', 'details' を返す場合
            # もしKaldiからのレスポンスが異なる構造の場合、ここで変換ロジックを追加する必要があります。
            # 例: raw_response['result']['final_score'] のようなネストされた構造
            # 今回は、raw_responseが直接 AnalysisResult.from_json に適合する形だと仮定します。
            
            analysis_result = AnalysisResult.from_json(raw_response)
            
            # ここでさらにビジネスロジックに基づいてスコアやフィードバックを調整することも可能
            # 例: _calculate_score(raw_response) などのメソッドは、JSONから取得した値を元に調整する場合に使用
            
            if not analysis_result.success:
                self._log(f'Analysis result indicates failure: {analysis_result.error_message}', is_error=True)
                return AnalysisResult.error(f'Analysis was not successful: {analysis_result.error_message}')
            
            if analysis_result.details is not None and not analysis_result.details.is_valid():
                self._log(f'Analysis details are out of valid range: {analysis_result.details}', is_error=True)
                # スコアや詳細の範囲外チェックが真であればエラーとして処理
                return AnalysisResult.error('Analysis details contain invalid values.')
            
            self._log('Analysis result processed successfully.')
            return analysis_result
            
        except ValueError as e:
            self._log(f'Failed to parse analysis result from Kaldi service: {e}', is_error=True)
            return AnalysisResult.error(f'Invalid analysis result format from service: {e}', original_error=e)
        
        except Exception as e:
            self._log(f'Error processing analysis result: {e}', is_error=True)
            return AnalysisResult.error(f'Error processing analysis result: {e}', original_error=e)


# 使用例
async def main():
    service = AnalyzeService()
    
    try:
        # 通常の分析
        result = await service.analyze_audio("path/to/audio.wav")
        print(result)
        
        # リトライ付き分析
        result_with_retry = await service.analyze_audio_with_retry("path/to/audio.wav", max_retries=5)
        print(result_with_retry)
        
    finally:
        service.dispose()


if __name__ == "__main__":
    asyncio.run(main())