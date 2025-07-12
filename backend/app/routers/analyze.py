import asyncio
import os
import tempfile
import logging
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass

import magic
import numpy as np
import re
import librosa
import soundfile as sf
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# PaddleSpeech関連のimport
PADDLESPEECH_AVAILABLE = False
try:
    from paddlespeech.cli.asr.infer import ASRExecutor
    from paddlespeech.cli.text.infer import TextExecutor
    from paddlespeech.cli.align.infer import AlignExecutor
    PADDLESPEECH_AVAILABLE = True
except ImportError as e:
    print(f"PaddleSpeech not available: {e}. Install with: pip install paddlespeech")

    class DummyASRExecutor:
        def __call__(self, audio_file, **kwargs):
            logging.warning("Dummy ASR Executor called.")
            return {"text": "dummy recognition result", "confidence": 0.5}
    class DummyTextExecutor:
        def __call__(self, text, **kwargs):
            logging.warning("Dummy Text Executor called.")
            # 簡易的なピンイン変換 (実際はより複雑)
            pinyin_map = {"你": "ni3", "好": "hao3", "中": "zhong1", "国": "guo2", "我": "wo3", "爱": "ai4", "你": "ni3"}
            return " ".join(pinyin_map.get(char, 'unknown') for char in text)
    class DummyAlignExecutor:
        def __call__(self, audio_file, text, **kwargs):
            logging.warning("Dummy Align Executor called.")
            duration = 5.0 # ダミーの音声長
            phonemes = []
            char_duration = duration / len(text) if text else 0
            current_time = 0.0
            for char in text:
                phonemes.append({"phone": char, "start": current_time, "end": current_time + char_duration, "score": 0.7})
                current_time += char_duration
            # 修正点: DummyAlignExecutorが直接リストを返すように変更
            return phonemes

    ASRExecutor = DummyASRExecutor
    TextExecutor = DummyTextExecutor
    AlignExecutor = DummyAlignExecutor


# --- 設定 ---
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg"]
MAX_FILE_SIZE_MB = 50
TIMEOUT_DURATION = 120 # 秒
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0 # 秒

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 例外クラス ---
class AudioAnalysisException(Exception): pass
class AudioValidationException(AudioAnalysisException): pass
class PaddleSpeechServiceException(AudioAnalysisException):
    def __init__(self, message: str, status_code: Optional[int] = None, original_error: Optional[Exception] = None):
        self.status_code = status_code
        super().__init__(message, original_error)

class PronunciationAnalysisError(Exception): pass
class PaddleSpeechError(PronunciationAnalysisError): pass
class AudioProcessingError(PronunciationAnalysisError): pass
class ValidationError(PronunciationAnalysisError): pass

# --- データクラス ---
@dataclass
class AnalysisDetails:
    pronunciation: float
    intonation: float
    rhythm: float

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'AnalysisDetails':
        return cls(pronunciation=float(data['pronunciation']), intonation=float(data['intonation']), rhythm=float(data['rhythm']))

    def is_valid(self) -> bool:
        return all(0 <= score <= 100 for score in [self.pronunciation, self.intonation, self.rhythm])

@dataclass
class AnalysisResult:
    success: bool
    score: Optional[float] = None
    feedback: Optional[str] = None
    details: Optional[AnalysisDetails] = None
    error_message: Optional[str] = None
    recognized_text: Optional[str] = None

    @classmethod
    def error(cls, message: str, original_error: Optional[Exception] = None, recognized_text: Optional[str] = None) -> 'AnalysisResult':
        error_msg = f"{message} ({original_error})" if original_error else message
        return cls(success=False, error_message=error_msg, recognized_text=recognized_text)

# --- 音素関連定数 ---
INITIALS = sorted(['zh', 'ch', 'sh', 'ng', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                   'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w'],
                  key=len, reverse=True)
FINALS = sorted(['iang', 'iong', 'uang', 'ueng', 'ian', 'iao', 'ing', 'ang',
                 'eng', 'ong', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'in', 'un',
                 'iu', 'ie', 'ui', 'ue', 'er', 'a', 'o', 'e', 'i', 'u', 'v'],
                key=len, reverse=True)

# --- PaddleSpeech設定クラス ---
@dataclass
class PaddleSpeechConfig:
    """PaddleSpeech設定"""
    asr_model: str = "conformer_wenetspeech"  # 中国語ASRモデル
    asr_lang: str = "zh"
    asr_sample_rate: int = 16000
    asr_config: Optional[str] = None
    asr_ckpt: Optional[str] = None
    align_model: str = "conformer_wenetspeech" # AlignExecutor用
    align_lang: str = "zh"
    align_sample_rate: int = 16000
    device: str = "cpu"  # or "gpu"
    enable_auto_log: bool = False

@dataclass
class AnalysisConfig:
    """分析設定のデータクラス"""
    max_text_length: int = 600
    min_text_length: int = 1
    audio_sample_rate: int = 16000
    tone_weight: float = 0.4
    pronunciation_weight: float = 0.6
    paddlespeech_config: Optional[PaddleSpeechConfig] = None

    def __post_init__(self):
        if self.paddlespeech_config is None:
            self.paddlespeech_config = PaddleSpeechConfig()

# グローバルな設定インスタンス (validate_inputsで使用するため)
config = AnalysisConfig()

# --- 入力検証関数 (グローバル) ---
def validate_inputs(audio_content: bytes, text: str, pinyin_list: List[str]) -> None:
    """入力データの検証を行う"""
    if not audio_content: raise ValidationError("音声データが空です")
    if not text or not text.strip(): raise ValidationError("テキストが空です")

    text_length = len(text.strip())
    if not (config.min_text_length <= text_length <= config.max_text_length):
        raise ValidationError(
            f"テキスト長が範囲外です。現在: {text_length}, "
            f"許可範囲: {config.min_text_length}-{config.max_text_length}"
        )

    if not pinyin_list: raise ValidationError("ピンインリストが空です")

def validate_audio_file_content(audio_file: UploadFile, content: bytes) -> None:
    """音声ファイルのコンテンツを検証"""
    if len(content) == 0: raise AudioValidationException("音声ファイルが空です。")
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024: raise AudioValidationException(f"音声ファイルが大きすぎます (最大: {MAX_FILE_SIZE_MB}MB)。")
    if audio_file.content_type and audio_file.content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"未サポートのMIMEタイプ: {audio_file.content_type}")

def validate_audio_file(audio_path: str) -> Tuple[bool, str]:
    """音声ファイルの検証"""
    try:
        if not Path(audio_path).exists(): return False, "音声ファイルが存在しません"
        y, sr = librosa.load(audio_path, sr=None) # 元のサンプリングレートでロード
        if len(y) == 0: return False, "音声データが空です"
        duration = len(y) / sr
        if duration < 0.1: return False, "音声が短すぎます（0.1秒未満）"
        if duration > 300.0: return False, "音声が長すぎます（300秒超）"
        return True, "OK"
    except Exception as e:
        return False, f"音声ファイル検証エラー: {str(e)}"

# --- 評価関数 (グローバル) ---
def _evaluate_tone_advanced(pinyin_with_tones: List[str],
                           aligned_words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """改良された声調評価"""
    tone_scores = []
    tone_details = []

    for i, pinyin in enumerate(pinyin_with_tones):
        if i < len(aligned_words_data):
            word_tone_evaluation = _evaluate_single_tone(pinyin, aligned_words_data[i])
            tone_scores.append(word_tone_evaluation["score"])
            tone_details.append(word_tone_evaluation)
        else:
            logger.warning(f"アラインメントデータがピンインリストに不足しています (インデックス: {i})")
            tone_details.append({
                "pinyin": pinyin,
                "expected_tone": int(re.search(r'[1-5]', pinyin).group()) if re.search(r'[1-5]', pinyin) else 0,
                "score": 0.3,
                "confidence": 0.0,
                "pitch_pattern": "data_missing"
            })
            tone_scores.append(0.3)

    overall_tone_score = np.mean(tone_scores) if tone_scores else 0.0
    suggestions = _generate_tone_suggestions(tone_details)

    return {
        "word_tone_scores": tone_details,
        "overall_tone_score": round(overall_tone_score, 3),
        "tone_distribution": _analyze_tone_distribution(tone_details),
        "improvement_suggestions": suggestions
    }

def _evaluate_single_tone(pinyin: str, word_data: Dict) -> Dict[str, Any]:
    """単一単語の声調評価"""
    expected_tone = re.search(r'[1-5]', pinyin)
    expected_tone_num = int(expected_tone.group()) if expected_tone else 0

    phones = word_data.get("phones", [])
    pitch_values = [p.get("pitch", 0) for p in phones if p.get("pitch") is not None]

    pitch_pattern_analysis = _analyze_pitch_pattern(pitch_values, expected_tone_num)

    tone_confidence = np.mean([p.get("score", 0) for p in phones]) if phones else 0.0

    score = tone_confidence
    if "correct" in pitch_pattern_analysis: score = min(1.0, score * 1.2)
    elif "incorrect" in pitch_pattern_analysis: score = max(0.2, score * 0.7)
    elif "insufficient" in pitch_pattern_analysis or "unclear" in pitch_pattern_analysis: score = max(0.3, score * 0.9)

    return {
        "pinyin": pinyin,
        "expected_tone": expected_tone_num,
        "score": round(score, 3),
        "confidence": round(tone_confidence, 3),
        "pitch_pattern": pitch_pattern_analysis
    }

def _analyze_pitch_pattern(pitch_values: List[float], expected_tone: int) -> str:
    """ピッチパターンの分析"""
    if len(pitch_values) < 2: return "insufficient_data"
    pitch_values_filtered = [p for p in pitch_values if p > 0]
    if len(pitch_values_filtered) < 2: return "insufficient_valid_data"

    mean_pitch = np.mean(pitch_values_filtered)
    std_dev_pitch = np.std(pitch_values_filtered)
    pitch_start = pitch_values_filtered[0]
    pitch_end = pitch_values_filtered[-1]
    pitch_range = max(pitch_values_filtered) - min(pitch_values_filtered)

    if expected_tone == 1:
        if mean_pitch > 150 and std_dev_pitch < 20 and pitch_range < 50: return "correct_flat"
        else: return "incorrect_unstable"
    elif expected_tone == 2:
        if pitch_end - pitch_start > 30 and std_dev_pitch > 15 and pitch_values_filtered[-1] > pitch_values_filtered[0]: return "correct_rising"
        else: return "incorrect_pattern"
    elif expected_tone == 3:
        min_idx = np.argmin(pitch_values_filtered)
        if min_idx > 0 and min_idx < len(pitch_values_filtered) - 1:
            if pitch_values_filtered[min_idx] < pitch_values_filtered[0] and \
               pitch_values_filtered[min_idx] < pitch_values_filtered[-1] and \
               (pitch_values_filtered[-1] - pitch_values_filtered[min_idx]) > 10:
                return "correct_dip"
        return "incorrect_pattern"
    elif expected_tone == 4:
        if pitch_start - pitch_end > 30 and std_dev_pitch > 15 and pitch_values_filtered[0] > pitch_values_filtered[-1]: return "correct_falling"
        else: return "incorrect_pattern"
    elif expected_tone == 5: return "neutral_tone"
    return "pattern_unclear"

def _analyze_tone_distribution(tone_details: List[Dict]) -> Dict[str, Any]:
    """声調分布の分析"""
    expected_tones = [t["expected_tone"] for t in tone_details]
    tone_counts = {i: expected_tones.count(i) for i in range(1, 6)}

    tone_avg_scores = {}
    for tone_num in range(1, 6):
        relevant_scores = [t["score"] for t in tone_details if t["expected_tone"] == tone_num]
        tone_avg_scores[tone_num] = round(np.mean(relevant_scores), 3) if relevant_scores else 0.0

    most_difficult_tone = None
    if tone_avg_scores:
        non_zero_scores = {k: v for k, v in tone_avg_scores.items() if v > 0}
        if non_zero_scores: most_difficult_tone = min(non_zero_scores.items(), key=lambda x: x[1])[0]
        else: most_difficult_tone = max(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None

    best_tone = None
    if tone_avg_scores:
        non_zero_scores = {k: v for k, v in tone_avg_scores.items() if v > 0}
        if non_zero_scores: best_tone = max(non_zero_scores.items(), key=lambda x: x[1])[0]
        else: best_tone = min(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None

    return {
        "tone_counts": tone_counts,
        "tone_average_scores": tone_avg_scores,
        "most_difficult_tone": most_difficult_tone,
        "best_tone": best_tone
    }

def _generate_tone_suggestions(tone_details: List[Dict]) -> List[str]:
    """声調改善提案の生成"""
    suggestions = []
    for detail in tone_details:
        score = detail["score"]
        expected_tone = detail["expected_tone"]
        pitch_pattern = detail["pitch_pattern"]
        if score < 0.6:
            pinyin_str = detail['pinyin']
            if expected_tone == 1:
                if "unstable" in pitch_pattern: suggestions.append(f"'{pinyin_str}' の第1声: ピッチをより平坦に保つように意識しましょう。音程の変動を抑えてください。")
                elif "incorrect" in pitch_pattern: suggestions.append(f"'{pinyin_str}' の第1声: 高く、平らに発音する練習をしましょう。")
            elif expected_tone == 2:
                if "falling" in pitch_pattern or "incorrect" in pitch_pattern: suggestions.append(f"'{pinyin_str}' の第2声: 音程をはっきりと上昇させるように練習しましょう。質問のイントネーションをイメージしてください。")
            elif expected_tone == 3:
                if "incorrect" in pitch_pattern: suggestions.append(f"'{pinyin_str}' の第3声: 音程を一度下げてから再び上げる「谷型」のパターンを練習しましょう。自然なディップを意識してください。")
            elif expected_tone == 4:
                if "rising" in pitch_pattern or "incorrect" in pitch_pattern: suggestions.append(f"'{pinyin_str}' の第4声: 音程を上から下へ一気に下降させるように発音しましょう。強く断定するイメージです。")
            elif expected_tone == 5: suggestions.append(f"'{pinyin_str}' の軽声: 短く、軽く、自然に発音するように意識しましょう。")
            else: suggestions.append(f"'{pinyin_str}' の声調: 発音の明瞭度を上げるために、ピッチの安定性や音素の明確さを意識しましょう。")
    return list(sorted(set(suggestions)))

# --- コア分析サービス (PaddleSpeechService) ---
class PaddleSpeechService:
    """PaddleSpeech統合サービス"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.ps_config = config.paddlespeech_config
        self._asr_executor = None
        self._text_executor = None
        self._align_executor = None

        self._executors_initialized = False

        self._pinyin_map = {
            '你': 'ni3', '好': 'hao3', '今': 'jin1', '天': 'tian1',
            '怎': 'zen3', '么': 'me', '样': 'yang4', '吗': 'ma',
            '我': 'wo3', '是': 'shi4', '在': 'zai4', '这': 'zhe4',
            '那': 'na4', '有': 'you3', '没': 'mei2', '的': 'de5',
            '了': 'le5', '都': 'dou1', '也': 'ye3', '和': 'he2',
            '？': '', '。': '', '！': ''
        }

    async def _initialize_executors_async(self):
        """PaddleSpeech実行器の非同期初期化"""
        if self._executors_initialized:
            return

        if not PADDLESPEECH_AVAILABLE:
            logger.warning("PaddleSpeech is not available. Using dummy executors.")
            self._asr_executor = ASRExecutor()
            self._text_executor = TextExecutor()
            self._align_executor = AlignExecutor()
            self._executors_initialized = True
            return

        try:
            logger.info("PaddleSpeech executors initializing...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._sync_initialize_executors)
            self._executors_initialized = True
            logger.info("PaddleSpeech executors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleSpeech executors: {e}", exc_info=True)
            raise PaddleSpeechError(f"PaddleSpeech initialization failed: {e}")

    def _sync_initialize_executors(self):
        """PaddleSpeech実行器の同期初期化 (run_in_executorから呼び出される)"""
        self._asr_executor = ASRExecutor()
        self._text_executor = TextExecutor()
        self._align_executor = AlignExecutor()

    def _extract_acoustic_features(self, audio_path: str) -> Dict[str, Any]:
        """音響特徴量の抽出"""
        try:
            y, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)
            if len(y) == 0: raise AudioProcessingError(f"音声データが空です: {audio_path}")

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.01)
            pitch_values = []
            if pitches.shape[1] > 0:
                for t in range(pitches.shape[1]):
                    if magnitudes[:, t].sum() > 0:
                        index = np.argmax(magnitudes[:, t])
                        pitch = pitches[index, t]
                        if pitch > 0: pitch_values.append(pitch)
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2)) if len(y) > 0 else 0

            return {
                "pitch_values": pitch_values,
                "mean_pitch": np.mean(pitch_values) if pitch_values else 0,
                "pitch_std": np.std(pitch_values) if pitch_values else 0,
                "duration": duration,
                "rms_energy": rms_energy,
            }
        except Exception as e:
            logger.error(f"音響特徴量抽出エラー ({audio_path}): {e}", exc_info=True)
            raise AudioProcessingError(f"音響特徴量抽出に失敗しました: {e}")

    async def _run_paddlespeech_asr(self, audio_path: str) -> Dict[str, Any]:
        """PaddleSpeech ASRを実行"""
        await self._initialize_executors_async()
        try:
            logger.info(f"Running PaddleSpeech ASR on: {audio_path}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._asr_executor(
                audio_file=audio_path,
                model=self.ps_config.asr_model,
                lang=self.ps_config.asr_lang,
                sample_rate=self.ps_config.asr_sample_rate,
                config=self.ps_config.asr_config,
                ckpt_path=self.ps_config.asr_ckpt,
                device=self.ps_config.device
            ))

            if isinstance(result, str):
                return {"text": result, "confidence": 0.8, "segments": []}
            elif isinstance(result, dict):
                return {"text": result.get("text", ""), "confidence": result.get("confidence", 0.8), "segments": result.get("segments", [])}
            elif hasattr(result, 'text'):
                return {"text": getattr(result, 'text', ''), "confidence": getattr(result, 'confidence', 0.8), "segments": getattr(result, 'segments', [])}
            else:
                logger.warning(f"Unexpected ASR Executor result format: {type(result)}")
                return {"text": "", "confidence": 0.0, "segments": []}
        except Exception as e:
            logger.error(f"PaddleSpeech ASR error: {e}", exc_info=True)
            raise PaddleSpeechError(f"ASR execution failed: {e}")

    async def _run_paddlespeech_alignment(self, audio_path: str, text: str) -> Dict[str, Any]:
        """PaddleSpeech強制アラインメントを実行"""
        await self._initialize_executors_async()
        try:
            logger.info(f"Running PaddleSpeech forced alignment on: {audio_path} with text: {text}")
            loop = asyncio.get_running_loop()
            align_output = await loop.run_in_executor(None, lambda: self._align_executor(
                audio_file=audio_path,
                text=text,
                model=self.ps_config.align_model,
                lang=self.ps_config.align_lang,
                sample_rate=self.ps_config.align_sample_rate,
                device=self.ps_config.device
            ))

            phones_data = []
            # 修正点: AlignExecutorの戻り値の型に対応
            if isinstance(align_output, list):
                # 直接リストの場合 (例: DummyAlignExecutorが直接phonemesリストを返すように修正後)
                data_to_process = align_output
            elif isinstance(align_output, dict) and 'result' in align_output and isinstance(align_output['result'], list):
                # 辞書で 'result' キーを持ち、その値がリストの場合 (例: 実際のAlignExecutorのデフォルト出力)
                data_to_process = align_output['result']
            elif hasattr(align_output, 'result') and isinstance(align_output.result, list):
                # result属性がリストの場合 (別の形式のAlignExecutor戻り値)
                data_to_process = align_output.result
            else:
                logger.warning(f"Unexpected AlignExecutor result format: {type(align_output)} - Content: {align_output}")
                raise PaddleSpeechError("AlignExecutor returned an unexpected format.")

            for item in data_to_process:
                if isinstance(item, dict) and 'phone' in item and 'start' in item and 'end' in item:
                    phones_data.append({"phone": item["phone"], "start": item["start"], "end": item["end"], "score": item.get("score", 0.8)})
                else:
                    logger.warning(f"Unexpected item format in alignment result list: {item}")


            logger.info(f"PaddleSpeech forced alignment completed. Found {len(phones_data)} phones.")
            return {"phones": phones_data}
        except Exception as e:
            logger.error(f"PaddleSpeech alignment error: {e}", exc_info=True)
            raise PaddleSpeechError(f"Alignment execution failed: {e}")

    async def _text_to_phonemes_g2p(self, text: str) -> List[str]:
        """テキストを音素に変換 (PaddleSpeech TextExecutorを使用)"""
        await self._initialize_executors_async()
        if not PADDLESPEECH_AVAILABLE or isinstance(self._text_executor, DummyTextExecutor):
            logger.warning("PaddleSpeech TextExecutor is not available or is dummy. Using fallback for G2P.")
            return self._text_to_phonemes_fallback(text)
        try:
            loop = asyncio.get_running_loop()
            phonemes_str = await loop.run_in_executor(None, lambda: self._text_executor(text=text, lang="zh", to_pinyin=True))

            pinyin_parts = phonemes_str.split()
            all_phonemes = []
            for pinyin_part in pinyin_parts:
                all_phonemes.extend(self._pinyin_to_phonemes(pinyin_part))
            return all_phonemes
        except Exception as e:
            logger.error(f"PaddleSpeech Text to phonemes conversion error: {e}", exc_info=True)
            return self._text_to_phonemes_fallback(text)

    def _text_to_phonemes_fallback(self, text: str) -> List[str]:
        """テキストを音素に変換（TextExecutorが利用できない場合のフォールバック）"""
        phonemes = []
        for char in text:
            pinyin = self._estimate_pinyin(char)
            char_phonemes = self._pinyin_to_phonemes(pinyin)
            phonemes.extend(char_phonemes)
        return phonemes

    def _estimate_pinyin(self, char: str) -> str:
        """文字からピンインを推定 (簡易版、_pinyin_mapを使用)"""
        return self._pinyin_map.get(char, "unknown")

    def _pinyin_to_phonemes(self, pinyin: str) -> List[str]:
        """ピンインを声母/韻母/声調に分解"""
        if pinyin == "unknown": return ["unknown"]
        clean_pinyin = re.sub(r'[1-5]', '', pinyin)
        tone = re.search(r'[1-5]', pinyin)
        tone_num = tone.group() if tone else "0"
        phonemes = []
        remaining = clean_pinyin
        for initial in INITIALS:
            if remaining.startswith(initial):
                phonemes.append(initial)
                remaining = remaining[len(initial):]
                break
        if remaining:
            for final in FINALS:
                if remaining.startswith(final):
                    phonemes.append(final)
                    remaining = remaining[len(final):]
                    break
        if tone_num != "0": phonemes.append(tone_num)
        return phonemes if phonemes else ["unknown"]

    def _calculate_score_from_pitch(self, pitch: float) -> float:
        """ピッチから音素スコアを計算"""
        if pitch == 0: return 0.5
        if 80 <= pitch <= 400: return 0.9
        elif 50 <= pitch < 80 or 400 < pitch <= 500: return 0.7
        else: return 0.4

    def _build_word_alignments(self, text: str, pinyin_list: List[str], phones: List[Dict]) -> List[Dict]:
        """音素アラインメントから単語アラインメントを構築"""
        word_alignments = []
        phone_idx = 0

        for char_idx, char in enumerate(text):
            current_pinyin = pinyin_list[char_idx] if char_idx < len(pinyin_list) else ""
            if not current_pinyin:
                logger.warning(f"テキスト '{char}' ({char_idx}) に対応するピンインが見つかりません。スキップします。")
                continue

            expected_phonemes = self._pinyin_to_phonemes(current_pinyin)
            num_expected_phones = len(expected_phonemes)

            word_phones = []
            start_time = phones[phone_idx]['start'] if phone_idx < len(phones) else 0.0

            for _ in range(num_expected_phones):
                if phone_idx < len(phones):
                    word_phones.append(phones[phone_idx])
                    phone_idx += 1
                else:
                    logger.warning(f"音素アラインメントが不足しています。テキスト: '{char}', ピンイン: '{current_pinyin}'")
                    break

            end_time = word_phones[-1]['end'] if word_phones else start_time
            word_score = np.mean([p['score'] for p in word_phones]) if word_phones else 0.0

            word_alignments.append({
                "word": char,
                "pinyin": current_pinyin,
                "start": start_time,
                "end": end_time,
                "phones": word_phones,
                "score": word_score
            })
        return word_alignments


    async def run_analysis(self, audio_content: bytes, text: str, pinyin_list: List[str]) -> Dict[str, Any]:
        """発音分析のメイン関数"""
        validate_inputs(audio_content, text, pinyin_list)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            audio_path = temp_dir / "input_audio.wav"

            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_content)
                logger.info(f"音声ファイルを保存: {audio_path}")

                is_valid, msg = validate_audio_file(str(audio_path))
                if not is_valid:
                    raise ValidationError(f"音声ファイルの検証エラー: {msg}")

                acoustic_features = self._extract_acoustic_features(str(audio_path))
                logger.info("音響特徴量抽出完了")

                asr_result = await self._run_paddlespeech_asr(str(audio_path))
                logger.info("PaddleSpeech ASR完了")

                alignment_result = await self._run_paddlespeech_alignment(str(audio_path), text)
                logger.info("強制アラインメント完了")

                final_aligned_phones = []
                pitch_values = acoustic_features.get("pitch_values", [])
                audio_duration = acoustic_features.get("duration", 1.0)

                for p in alignment_result["phones"]:
                    start_sec = p["start"]
                    end_sec = p["end"]

                    segment_mean_pitch = 0
                    if pitch_values and audio_duration > 0:
                        frame_rate = len(pitch_values) / audio_duration
                        start_idx = int(start_sec * frame_rate)
                        end_idx = int(end_sec * frame_rate)
                        segment_pitches = [pv for i, pv in enumerate(pitch_values) if start_idx <= i < end_idx and pv > 0]
                        segment_mean_pitch = np.mean(segment_pitches) if segment_pitches else 0

                    p["pitch"] = segment_mean_pitch

                    original_align_score = p.get("score", 0.7)
                    pitch_score_contribution = self._calculate_score_from_pitch(p["pitch"])

                    p["score"] = (original_align_score * 0.7 + pitch_score_contribution * 0.3)
                    p["score"] = max(0.2, min(1.0, p["score"]))
                    final_aligned_phones.append(p)

                if final_aligned_phones:
                    word_alignments = self._build_word_alignments(text, pinyin_list, final_aligned_phones)

                    tone_evaluation = _evaluate_tone_advanced(pinyin_list, word_alignments)

                    pronunciation_scores = [p["score"] for w in word_alignments for p in w["phones"]]
                    overall_pronunciation_score = np.mean(pronunciation_scores) if pronunciation_scores else 0.0

                    final_score = (
                        tone_evaluation["overall_tone_score"] * self.config.tone_weight +
                        overall_pronunciation_score * self.config.pronunciation_weight
                    )

                    logger.info("分析完了")

                    return {
                        "overall_score": round(final_score, 3),
                        "overall_pronunciation_score": round(overall_pronunciation_score, 3),
                        "tone_evaluation": tone_evaluation,
                        "word_alignments": word_alignments,
                        "asr_result": asr_result,
                        "acoustic_features_summary": {
                            "duration": acoustic_features.get("duration"),
                            "mean_pitch": acoustic_features.get("mean_pitch"),
                            "pitch_std": acoustic_features.get("pitch_std"),
                            "rms_energy": acoustic_features.get("rms_energy")
                        }
                    }
                else:
                    raise PaddleSpeechError("アラインメント結果から有効な音素データが見つかりませんでした。")

            except (ValidationError, AudioProcessingError, PaddleSpeechError) as e:
                logger.error(f"分析処理中にエラーが発生しました: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.critical(f"予期せぬクリティカルエラーが発生しました: {e}", exc_info=True)
                raise PronunciationAnalysisError(f"分析処理中に予期せぬエラーが発生しました: {e}")

# --- FastAPI向け分析サービス (PronunciationAnalysisService) ---
class PronunciationAnalysisService:
    def __init__(self):
        self.core_paddlespeech_service = PaddleSpeechService(config)
        self.mime_magic = magic.Magic(mime=True)

    async def analyze_audio(self, audio_file_path: Path, text: str) -> AnalysisResult:
        logger.info(f'音声分析開始: {audio_file_path}, テキスト: {text}')
        try:
            pinyin_list_for_evaluation = await self._get_pinyin_list_for_evaluation(text)
            
            analysis_data = await self.core_paddlespeech_service.run_analysis(
                audio_file_path.read_bytes(), text, pinyin_list_for_evaluation
            )

            overall_score = analysis_data.get("overall_score", 0.0) * 100
            pronunciation_score = analysis_data.get("overall_pronunciation_score", 0.0) * 100
            tone_score = analysis_data.get("tone_evaluation", {}).get("overall_tone_score", 0.0) * 100

            feedback_messages = analysis_data.get("tone_evaluation", {}).get("improvement_suggestions", [])

            recognized_text = analysis_data.get("asr_result", {}).get("text", "").strip()
            if recognized_text and recognized_text != text.strip():
                feedback_messages.append(f"認識されたテキスト: '{recognized_text}'。元のテキストと一致しません。")

            feedback = " ".join(feedback_messages) if feedback_messages else "発音は良好です。"

            details = AnalysisDetails(
                pronunciation=pronunciation_score,
                intonation=tone_score,
                rhythm=0.0
            )

            return AnalysisResult(
                success=True,
                score=overall_score,
                feedback=feedback,
                details=details,
                recognized_text=recognized_text
            )

        except (AudioValidationException, PaddleSpeechError, PronunciationAnalysisError, ValidationError) as e:
            logger.error(f'分析エラー: {e}', exc_info=True)
            return AnalysisResult.error(f'分析エラー: {e}', e)
        except Exception as e:
            logger.exception('予期せぬエラー発生')
            return AnalysisResult.error(f'処理エラー: {str(e)}', e)

    async def analyze_with_retry(self, audio_file_path: Path, text: str) -> AnalysisResult:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return await self.analyze_audio(audio_file_path, text)
            except PaddleSpeechServiceException as e:
                if attempt == RETRY_ATTEMPTS - 1: raise
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                logger.info(f'リトライ {attempt + 1}/{RETRY_ATTEMPTS}')
        raise PaddleSpeechServiceException('最大リトライ回数を超過しました')

    async def _get_pinyin_list_for_evaluation(self, text: str) -> List[str]:
        """中国語テキストから評価用のピンインリストを生成"""
        await self.core_paddlespeech_service._initialize_executors_async()

        if PADDLESPEECH_AVAILABLE and not isinstance(self.core_paddlespeech_service._text_executor, DummyTextExecutor):
            try:
                loop = asyncio.get_running_loop()
                pinyin_str = await loop.run_in_executor(None, lambda: self.core_paddlespeech_service._text_executor(
                    text=text, lang="zh", to_pinyin=True
                ))
                pinyin_parts = pinyin_str.split()
                if len(text) == len(pinyin_parts):
                    return pinyin_parts
                else:
                    logger.warning(f"TextExecutorからのピンインと元のテキストの文字数が一致しません。テキスト: {len(text)}, ピンイン: {len(pinyin_parts)}. フォールバックします。")
                    return self._estimate_pinyin_list_from_text_fallback(text)
            except Exception as e:
                logger.error(f"TextExecutorでのピンイン生成に失敗しました: {e}", exc_info=True)
                return self._estimate_pinyin_list_from_text_fallback(text)
        else:
            logger.warning("PaddleSpeech TextExecutorが利用できないため、フォールバックでピンインを生成します。")
            return self._estimate_pinyin_list_from_text_fallback(text)

    def _estimate_pinyin_list_from_text_fallback(self, text: str) -> List[str]:
        """中国語テキストからピンインリストを生成 (フォールバック用)"""
        results = []
        for char in text:
            pinyin = self.core_paddlespeech_service._pinyin_map.get(char, '')
            if not pinyin and char.strip():
                logger.warning(f"ピンイン未登録の文字: {char} (フォールバック)")
            results.append(pinyin)
        return results

# --- グローバルサービスインスタンス ---
analyze_service = PronunciationAnalysisService()

# --- APIルーター ---
router = APIRouter(tags=["analyze"])

# --- 発音分析エンドポイント ---
@router.post("/")
async def analyze_pronunciation(
    audio: UploadFile = File(...),
    text: str = Form(...),
):
    logger.info(f"分析リクエスト受信 - テキスト: '{text}'")
    temp_file_path = None
    try:
        content = await audio.read()
        validate_audio_file_content(audio, content)

        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        if not suffix.startswith('.'): suffix = '.' + suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as temp_audio:
            temp_audio.write(content)
            temp_file_path = Path(temp_audio.name)

        logger.info(f"音声ファイル保存: {temp_file_path}, サイズ: {len(content)} bytes")

        result = await analyze_service.analyze_with_retry(temp_file_path, text)

        if not result.success:
            logger.error(f"分析サービスからのエラー応答: {result.error_message}")
            raise HTTPException(status_code=400, detail=result.error_message)

        response_content = {
            "success": result.success,
            "score": result.score,
            "feedback": result.feedback,
            "recognized_text": result.recognized_text
        }

        if result.details:
            response_content["details"] = {
                "pronunciation": result.details.pronunciation,
                "intonation": result.details.intonation,
                "rhythm": result.details.rhythm
            }

        return JSONResponse(content=response_content)

    except AudioValidationException as e:
        logger.error(f"音声検証エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except PaddleSpeechServiceException as e:
        logger.error(f"PaddleSpeechサービスエラー: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValidationError as e:
        logger.error(f"入力検証エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"予期しないエラー: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部サーバーエラーが発生しました")
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                os.unlink(temp_file_path)
                logger.info(f"一時ファイル削除: {temp_file_path}")
            except OSError as e:
                logger.warning(f"一時ファイルの削除に失敗: {temp_file_path} - {e}")

# --- ヘルスチェックエンドポイント ---
@router.get("/health")
async def health_check():
    status = "ok" if PADDLESPEECH_AVAILABLE else "degraded"
    message = "PaddleSpeech is available." if PADDLESPEECH_AVAILABLE else "PaddleSpeech is not installed or failed to import. Using dummy executors."

    executors_ready = False
    if PADDLESPEECH_AVAILABLE: 
        try:
            # analyze_serviceがインスタンス化されていることを前提
            if analyze_service.core_paddlespeech_service._executors_initialized:
                executors_ready = True

        except Exception as e:
            logger.warning(f"PaddleSpeech executor status check failed: {e}")
            pass
    return {
        "status": status,
        "service": "pronunciation_analysis",
        "paddlespeech_available": PADDLESPEECH_AVAILABLE,
        "paddlespeech_executors_ready": executors_ready,
        "message": message
    }

# --- モデル一覧エンドポイント ---
@router.get("/models")
async def get_available_models():
    # PaddleSpeechの利用可能なASR/Alignモデルをリストアップ
    return {
        "models": [
            {"id": "conformer_wenetspeech", "name": "Conformer WenetSpeech (Chinese)", "description": "PaddleSpeechの推奨中国語ASRおよびアラインメントモデル"},
            {"id": "dummy", "name": "Dummy Model", "description": "PaddleSpeechが利用できない場合のフォールバックモデル"}
        ]
    }

# --- テスト用エンドポイント ---
@router.post("/test")
async def test_endpoint():
    return {"message": "Test endpoint is working", "status": "ok", "timestamp": str(datetime.now())}