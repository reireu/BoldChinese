import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .config import KaldiConfig
from ..utils.errors import AudioProcessingError

@dataclass
class AudioFeatures:
    """音声特徴量を保持するデータクラス"""
    path: Path
    features: Dict[str, Any]
    summary: Dict[str, Any]

class AudioProcessor:
    def __init__(self, config: KaldiConfig):
        self.config = config
        self.temp_dir = Path(tempfile.gettempdir()) / "boldchinese_audio"
        self.temp_dir.mkdir(exist_ok=True)

    async def process_audio(self, audio_content: bytes) -> AudioFeatures:
        """音声処理のメインメソッド"""
        try:
            temp_path = await self._save_audio_temp(audio_content)
            features = await self._extract_features(temp_path)
            return AudioFeatures(
                path=temp_path,
                features=features,
                summary=self._create_feature_summary(features)
            )
        except Exception as e:
            raise AudioProcessingError(f"音声処理失敗: {str(e)}") from e

    async def _save_audio_temp(self, audio_content: bytes) -> Path:
        """音声データを一時ファイルとして保存"""
        temp_path = self.temp_dir / f"audio_{id(audio_content)}.wav"
        temp_path.write_bytes(audio_content)
        return temp_path

    async def _extract_features(self, audio_path: Path) -> Dict[str, Any]:
        """音声特徴量の抽出"""
        y, sr = librosa.load(str(audio_path), sr=self.config.sample_rate)
        
        return {
            "pitch": self._extract_pitch(y, sr),
            "energy": self._extract_energy(y),
            "duration": len(y) / sr,
            "sample_rate": sr,
            "mfcc": self._extract_mfcc(y, sr)
        }

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """ピッチ特徴量の抽出"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        return {
            "values": pitches,
            "magnitudes": magnitudes,
            "statistics": {
                "mean": float(np.mean(pitches[pitches > 0])),
                "std": float(np.std(pitches[pitches > 0]))
            }
        }

    def _extract_energy(self, y: np.ndarray) -> Dict[str, Any]:
        """エネルギー特徴量の抽出"""
        rms = librosa.feature.rms(y=y)[0]
        return {
            "values": rms,
            "statistics": {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms))
            }
        }

    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """MFCC特徴量の抽出"""
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def _create_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """特徴量のサマリー生成"""
        return {
            "duration": features["duration"],
            "mean_pitch": features["pitch"]["statistics"]["mean"],
            "pitch_std": features["pitch"]["statistics"]["std"],
            "mean_energy": features["energy"]["statistics"]["mean"],
            "sample_rate": features["sample_rate"]
        }

    def __del__(self):
        """インスタンス破棄時に一時ファイルを削除"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            for file in self.temp_dir.glob("*.wav"):
                file.unlink(missing_ok=True)
            self.temp_dir.rmdir()