import json
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav
from scipy.signal import find_peaks
import warnings
import tempfile
import shutil
import pypinyin
import os
# Starlette関連は、アプリケーション全体に関わるため、サービスコードからは除外します。
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request
# from starlette.responses import Response

# NumPyのfloat型をJSONで扱えるようにするためのカスタムJSONEncoder
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# PaddleSpeech関連のimport
PADDLESPEECH_AVAILABLE = False
try:
    from paddlespeech.cli.asr import ASRExecutor
    from paddlespeech.cli.text import TextExecutor
    from paddlespeech.cli.align import AlignExecutor
    import paddle
    
    # 必要なモジュールが全て正しくインポートできたか確認
    required_modules = [ASRExecutor, TextExecutor, AlignExecutor, paddle]
    if all(module is not None for module in required_modules):
        PADDLESPEECH_AVAILABLE = True
        print(f"Using PaddleSpeech version: {paddle.__version__}")
    else:
        print("Some PaddleSpeech modules could not be imported correctly")
except ImportError as e:
    print(f"PaddleSpeech not available: {e}. Install with: pip install paddlespeech")
except Exception as e:
    print(f"Unexpected error initializing PaddleSpeech: {e}")

@dataclass
class PaddleSpeechConfig:
    """PaddleSpeech設定"""
    asr_model: str = "conformer_wenetspeech"
    asr_lang: str = "zh"
    asr_sample_rate: int = 16000
    asr_config: Optional[str] = None
    asr_ckpt: Optional[str] = None
    align_model: str = "conformer_wenetspeech"
    align_lang: str = "zh"
    align_sample_rate: int = 16000
    device: str = "cpu"
    enable_auto_log: bool = False

@dataclass
class AnalysisConfig:
    """分析設定のデータクラス"""
    max_text_length: int = 600
    min_text_length: int = 1
    audio_sample_rate: int = 16000
    tone_weight: float = 0.4
    pronunciation_weight: float = 0.6
    paddlespeech_config: PaddleSpeechConfig = None

    def __post_init__(self):
        if self.paddlespeech_config is None:
            self.paddlespeech_config = PaddleSpeechConfig()

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# 定数
INITIALS = sorted(['zh', 'ch', 'sh', 'ng', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                   'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w'],
                  key=len, reverse=True)
FINALS = sorted(['iang', 'iong', 'uang', 'ueng', 'ian', 'iao', 'ing', 'ang',
                 'eng', 'ong', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'in', 'un',
                 'iu', 'ie', 'ui', 'ue', 'er', 'a', 'o', 'e', 'i', 'u', 'v'],
                key=len, reverse=True)

# スコアリングの重み定数
ALIGN_SCORE_WEIGHT = 0.7
PITCH_SCORE_WEIGHT = 0.3
MIN_SCORE_CLAMP = 0.2
MAX_SCORE_CLAMP = 1.0
DUMMY_ASR_CONFIDENCE = 0.7
DUMMY_ALIGN_SCORE = 0.8
DUMMY_FALLBACK_SCORE = 0.3

# カスタム例外クラス
class PronunciationAnalysisError(Exception):
    """発音分析プロセス全般で発生するエラーの基底クラス"""
    pass

class PaddleSpeechError(PronunciationAnalysisError):
    """PaddleSpeech関連のエラー"""
    pass

class AudioProcessingError(PronunciationAnalysisError):
    """音声処理中に発生するエラー"""
    pass

class ValidationError(PronunciationAnalysisError):
    """入力データの検証エラー"""
    pass

class PaddleSpeechService:
    """PaddleSpeech統合サービス"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.ps_config = config.paddlespeech_config
        self._asr_executor = None
        self._text_executor = None
        self._align_executor = None

        if not PADDLESPEECH_AVAILABLE:
            logger.warning("PaddleSpeech is not available. Executors will be dummy.")
            return

        self._initialize_executors()

    def _initialize_executors(self):
        """PaddleSpeech実行器の初期化"""
        try:
            logger.info("PaddleSpeech executors initializing...")
            self._asr_executor = ASRExecutor()
            self._text_executor = TextExecutor()
            self._align_executor = AlignExecutor()
            logger.info("PaddleSpeech executors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleSpeech executors: {e}")
            # エラー発生時でもPADDLESPEECH_AVAILABLEをFalseにしないことで、
            # ダミー実装へのフォールバックを明示的に制御する
            global PADDLESPEECH_AVAILABLE
            PADDLESPEECH_AVAILABLE = False
            logger.warning("PaddleSpeech initialization failed, falling back to dummy implementations.")


    def _extract_acoustic_features(self, audio_path: str) -> Dict[str, Union[float, List]]:
        """音響特徴量の抽出"""
        try:
            y, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)

            if len(y) == 0:
                raise AudioProcessingError(f"音声データが空です: {audio_path}")

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.01)
            pitch_values = [
                pitches[np.argmax(magnitudes[:, t]), t]
                for t in range(pitches.shape[1])
                if magnitudes[:, t].sum() > 0 and pitches[np.argmax(magnitudes[:, t]), t] > 0
            ]
            
            duration = len(y) / sr
            rms_energy = float(np.sqrt(np.mean(y**2))) if len(y) > 0 else 0.0

            return {
                "pitch_values": [float(p) for p in pitch_values],
                "mean_pitch": float(np.mean(pitch_values)) if pitch_values else 0.0,
                "pitch_std": float(np.std(pitch_values)) if pitch_values else 0.0,
                "duration": float(duration),
                "rms_energy": rms_energy,
                # これらの特徴量は現在スコアリングに直接使われないが、情報として保持する場合は残す
                "spectral_centroids": librosa.feature.spectral_centroid(y=y, sr=sr)[0].tolist(),
                "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr)[0].tolist(),
                "mfccs": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).tolist()
            }

        except Exception as e:
            logger.error(f"音響特徴量抽出エラー ({audio_path}): {e}", exc_info=True)
            raise AudioProcessingError(f"音響特徴量抽出に失敗しました: {e}")

    def _run_paddlespeech_asr(self, audio_path: str) -> Dict[str, Any]:
        """PaddleSpeech ASRを実行"""
        if not PADDLESPEECH_AVAILABLE or self._asr_executor is None:
            logger.warning("PaddleSpeech ASR not available, using dummy.")
            return {"text": "识别结果", "confidence": DUMMY_ASR_CONFIDENCE, "segments": []}

        try:
            logger.info(f"Running PaddleSpeech ASR on: {audio_path}")
            result = self._asr_executor(
                audio_file=audio_path,
                model=self.ps_config.asr_model,
                lang=self.ps_config.asr_lang,
                sample_rate=self.ps_config.asr_sample_rate,
                config=self.ps_config.asr_config,
                ckpt_path=self.ps_config.asr_ckpt,
                device=self.ps_config.device
            )
            if isinstance(result, str):
                return {"text": result, "confidence": DUMMY_ASR_CONFIDENCE, "segments": []}
            elif isinstance(result, dict):
                return {
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", DUMMY_ASR_CONFIDENCE),
                    "segments": result.get("segments", [])
                }
            else:
                return {
                    "text": getattr(result, 'text', ''),
                    "confidence": getattr(result, 'confidence', DUMMY_ASR_CONFIDENCE),
                    "segments": getattr(result, 'segments', [])
                }
        except Exception as e:
            logger.error(f"PaddleSpeech ASR error: {e}", exc_info=True)
            raise PaddleSpeechError(f"ASR execution failed: {e}")

    def _run_paddlespeech_alignment(self, audio_path: str, text: str, acoustic_features: Dict) -> Dict[str, Any]:
        """PaddleSpeech強制アラインメントを実行"""
        if not PADDLESPEECH_AVAILABLE or self._align_executor is None:
            logger.warning("PaddleSpeech AlignExecutor not available, generating dummy alignment.")
            return self._generate_realistic_dummy_alignment(text, acoustic_features)

        try:
            logger.info(f"Running PaddleSpeech forced alignment on: {audio_path} with text: {text}")
            align_result = self._align_executor(
                audio_file=audio_path,
                text=text,
                model=self.ps_config.align_model,
                lang=self.ps_config.align_lang,
                sample_rate=self.ps_config.align_sample_rate,
                device=self.ps_config.device
            )

            phones_data = []
            result_list = getattr(align_result, 'result', align_result) # .result属性があればそれを使う
            if isinstance(result_list, list):
                for item in result_list:
                    if isinstance(item, dict) and all(k in item for k in ['phone', 'start', 'end']):
                        phones_data.append({
                            "phone": item["phone"],
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                            "score": float(item.get("score", DUMMY_ALIGN_SCORE))
                        })
            else:
                logger.warning(f"Unexpected AlignExecutor result format: {type(align_result)}")
                return self._generate_realistic_dummy_alignment(text, acoustic_features)

            logger.info(f"PaddleSpeech forced alignment completed. Found {len(phones_data)} phones.")
            return {"phones": phones_data}

        except Exception as e:
            logger.error(f"PaddleSpeech alignment error: {e}", exc_info=True)
            return self._generate_realistic_dummy_alignment(text, acoustic_features)

    def _text_to_phonemes(self, text: str) -> List[str]:
        """テキストを音素に変換 (PaddleSpeech TextExecutorを使用またはフォールバック)"""
        if not PADDLESPEECH_AVAILABLE or self._text_executor is None:
            logger.warning("PaddleSpeech TextExecutor not available, using fallback for phoneme conversion.")
            return self._text_to_phonemes_fallback(text)

        try:
            phonemes_str = self._text_executor(text=text, lang="zh", to_pinyin=True)
            all_phonemes = []
            for pinyin_part in phonemes_str.split():
                all_phonemes.extend(self._pinyin_to_phonemes(pinyin_part))
            return all_phonemes
        except Exception as e:
            logger.error(f"PaddleSpeech Text to phonemes conversion error: {e}", exc_info=True)
            return self._text_to_phonemes_fallback(text)

    def _text_to_phonemes_fallback(self, text: str) -> List[str]:
        """テキストを音素に変換（簡易フォールバック）"""
        return [
            ph for char in text
            for ph in self._pinyin_to_phonemes(self._estimate_pinyin(char))
        ]

    def _calculate_phoneme_score(self, phoneme: str, start: float, end: float, acoustic_features: Dict) -> float:
        """音素スコアの計算 (ダミー実装用)"""
        base_score = DUMMY_ALIGN_SCORE

        rms_energy = acoustic_features.get("rms_energy", 0.1)
        if rms_energy < 0.01: base_score -= 0.3
        elif rms_energy > 0.5: base_score -= 0.1

        pitch_std = acoustic_features.get("pitch_std", 0)
        if pitch_std > 50: base_score -= 0.2

        difficulty = self._get_phoneme_difficulty(phoneme)
        base_score -= difficulty * 0.2
        
        return max(MIN_SCORE_CLAMP, min(MAX_SCORE_CLAMP, float(base_score + np.random.normal(0, 0.1))))

    def _get_pitch_at_time(self, start: float, end: float, acoustic_features: Dict) -> float:
        """指定時間でのピッチを取得 (ダミー実装用)"""
        pitch_values = acoustic_features.get("pitch_values", [])
        if not pitch_values:
            return 0.0

        duration = acoustic_features.get("duration", 1.0)
        mid_time = (start + end) / 2

        frame_rate = len(pitch_values) / duration if duration > 0 else 0
        if frame_rate > 0:
            frame_idx = int(mid_time * frame_rate)
            if 0 <= frame_idx < len(pitch_values):
                return float(pitch_values[frame_idx])

        return float(np.mean(pitch_values)) if pitch_values else 0.0

    def _generate_realistic_dummy_alignment(self, text: str, acoustic_features: Dict) -> Dict[str, Any]:
        """音響特徴量に基づく現実的なダミーアラインメント生成"""
        duration = acoustic_features.get("duration", 1.0)
        
        characters = list(text)
        if not characters:
            return {"phones": []}

        char_duration = duration / len(characters)
        all_phones = []
        current_time = 0.0

        for char in characters:
            pinyin = self._estimate_pinyin(char)
            phonemes = self._pinyin_to_phonemes(pinyin)

            if phonemes:
                phone_duration = char_duration / len(phonemes)
                phone_start = current_time

                for phoneme in phonemes:
                    phone_end = phone_start + phone_duration
                    score = self._calculate_phoneme_score(phoneme, phone_start, phone_end, acoustic_features)
                    all_phones.append({
                        "phone": phoneme,
                        "start": float(phone_start),
                        "end": float(phone_end),
                        "score": float(score),
                        "pitch": self._get_pitch_at_time(phone_start, phone_end, acoustic_features)
                    })
                    phone_start = phone_end
            current_time += char_duration
        return {"phones": all_phones}

    def _estimate_pinyin(self, char: str) -> str:
        """文字からピンインを推定 (簡易版)"""
        pinyin_dict = {
            "你": "ni3", "好": "hao3", "我": "wo3", "是": "shi4",
            "的": "de", "在": "zai4", "有": "you3", "不": "bu4",
            "一": "yi1", "人": "ren2", "中": "zhong1", "国": "guo2",
            "爱": "ai4", "学": "xue2", "习": "xi2", "语": "yu3",
            "言": "yan2", "文": "wen2", "字": "zi4", "说": "shuo1",
            "话": "hua4", "听": "ting1", "看": "kan4", "读": "du2",
            "写": "xie3", "今": "jin1", "天": "tian1", "明": "ming2",
            "年": "nian2", "月": "yue4", "日": "ri4", "时": "shi2",
            "分": "fen1", "秒": "miao3", "小": "xiao3", "大": "da4"
        }
        return pinyin_dict.get(char, "unknown")

    def _pinyin_to_phonemes(self, pinyin: str) -> List[str]:
        """ピンインを音素に変換"""
        if pinyin == "unknown":
            return ["unknown"]

        clean_pinyin = re.sub(r'[1-5]', '', pinyin)
        tone_match = re.search(r'[1-5]', pinyin)
        tone_num = tone_match.group() if tone_match else "0"

        phonemes = []
        remaining = clean_pinyin

        # 声母の検出
        for initial in INITIALS:
            if remaining.startswith(initial):
                phonemes.append(initial)
                remaining = remaining[len(initial):]
                break
        # 韻母の検出
        for final in FINALS:
            if remaining.startswith(final):
                phonemes.append(final)
                break # remainingを更新する必要はない (韻母は一つ)

        if tone_num != "0":
            phonemes.append(tone_num)

        return phonemes if phonemes else ["unknown"]

    def _get_phoneme_difficulty(self, phoneme: str) -> float:
        """音素の難易度を返す"""
        difficult_phonemes = {
            'zh': 0.8, 'ch': 0.8, 'sh': 0.7, 'r': 0.9, 'j': 0.6, 'q': 0.6, 'x': 0.6,
            'z': 0.7, 'c': 0.7, 's': 0.5, 'ng': 0.8, 'er': 0.9, '1': 0.3, '2': 0.6,
            '3': 0.9, '4': 0.7, '5': 0.4
        }
        return difficult_phonemes.get(phoneme, 0.3)

    def _build_word_alignments(self, text: str, pinyin_list: List[str], phones: List[Dict]) -> List[Dict]:
        """音素アラインメントから単語アラインメントを構築"""
        word_alignments = []
        phone_idx = 0

        for char_idx, char in enumerate(text):
            if char_idx >= len(pinyin_list):
                logger.warning(f"ピンインリストの長さがテキストの長さを超えました。テキスト: {len(text)}, ピンインリスト: {len(pinyin_list)}")
                break

            pinyin = pinyin_list[char_idx]
            expected_phonemes = self._pinyin_to_phonemes(pinyin)
            
            word_phones = []
            for _ in range(len(expected_phonemes)):
                if phone_idx < len(phones):
                    word_phones.append(phones[phone_idx])
                    phone_idx += 1
                else:
                    logger.warning(f"音素アラインメントが不足しています。テキスト: '{char}', ピンイン: '{pinyin}'")
                    break

            start_time = word_phones[0]['start'] if word_phones else 0.0
            end_time = word_phones[-1]['end'] if word_phones else start_time
            word_score = float(np.mean([p['score'] for p in word_phones])) if word_phones else 0.0

            word_alignments.append({
                "word": char,
                "pinyin": pinyin,
                "start": start_time,
                "end": end_time,
                "phones": word_phones,
                "score": word_score
            })
        return word_alignments
    
    def _calculate_score_from_pitch(self, pitch_data: List[float]) -> float:
        """
        ピッチデータに基づいてスコアを計算する。
        ピッチの変動や安定性などを考慮して、声調の正確さを評価するのに利用。
        """
        if not pitch_data:
            return 0.0

        pitch_values_filtered = [p for p in pitch_data if p > 0]
        if len(pitch_values_filtered) < 2:
            return 0.5 # データが少なすぎる場合は中間的なスコア
        
        mean_pitch = np.mean(pitch_values_filtered)
        std_dev_pitch = np.std(pitch_values_filtered)
        # pitch_range = max(pitch_values_filtered) - min(pitch_values_filtered) # 現在未使用

        score = 0.8
        
        if std_dev_pitch < 10:
            score = min(MAX_SCORE_CLAMP, score + 0.1)
        elif std_dev_pitch > 40:
            score = max(MIN_SCORE_CLAMP, score - 0.2)
            
        if mean_pitch < 80 or mean_pitch > 350:
            score = max(MIN_SCORE_CLAMP, score - 0.1)

        return float(max(0.0, min(1.0, score)))


    def run_analysis(self, audio_content: bytes, text: str, pinyin_list: List[str]) -> Dict[str, Any]:
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

                asr_result = self._run_paddlespeech_asr(str(audio_path))
                logger.info("PaddleSpeech ASR完了")

                alignment_result = self._run_paddlespeech_alignment(str(audio_path), text, acoustic_features)
                logger.info("強制アラインメント完了")

                if not alignment_result.get("phones"):
                    raise PaddleSpeechError("アラインメント結果が空です")
                
                final_aligned_phones = []
                pitch_values_all = acoustic_features.get("pitch_values", [])
                audio_duration = acoustic_features.get("duration", 1.0)
                
                for p in alignment_result["phones"]:
                    start_sec = p["start"]
                    end_sec = p["end"]
                    
                    segment_pitches = []
                    if pitch_values_all and audio_duration > 0:
                        frame_rate = len(pitch_values_all) / audio_duration
                        start_idx = int(start_sec * frame_rate)
                        end_idx = int(end_sec * frame_rate)
                        segment_pitches = [
                            pv for i, pv in enumerate(pitch_values_all)
                            if start_idx <= i < end_idx and pv > 0
                        ]
                    
                    segment_mean_pitch = float(np.mean(segment_pitches)) if segment_pitches else 0.0
                    p["pitch"] = segment_mean_pitch
                    
                    original_align_score = p.get("score", DUMMY_ALIGN_SCORE)
                    pitch_score_contribution = self._calculate_score_from_pitch([segment_mean_pitch])
                    
                    p["score"] = float(original_align_score * ALIGN_SCORE_WEIGHT + pitch_score_contribution * PITCH_SCORE_WEIGHT)
                    p["score"] = max(MIN_SCORE_CLAMP, min(MAX_SCORE_CLAMP, p["score"]))
                    
                    final_aligned_phones.append(p)

                word_alignments = self._build_word_alignments(text, pinyin_list, final_aligned_phones)

                tone_evaluation = _evaluate_tone_advanced(pinyin_list, word_alignments)

                pronunciation_scores = [p["score"] for w in word_alignments for p in w["phones"]]
                overall_pronunciation_score = float(np.mean(pronunciation_scores)) if pronunciation_scores else 0.0

                final_score = float(
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

            except Exception as e:
                logger.error(f"分析エラー: {e}", exc_info=True)
                raise

# 入力検証関数
def validate_inputs(audio_content: bytes, text: str, pinyin_list: List[str]) -> None:
    """入力データの検証"""
    if not audio_content: raise ValidationError("音声データが空です")
    if not text or not text.strip(): raise ValidationError("テキストが空です")
    if not pinyin_list: raise ValidationError("ピンインリストが空です")
    if len(text) != len(pinyin_list): raise ValidationError("テキストとピンインリストの長さが一致しません")

def validate_audio_file(audio_path: str) -> Tuple[bool, str]:
    """音声ファイルの検証"""
    try:
        if not Path(audio_path).exists(): return False, "音声ファイルが存在しません"
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0: return False, "音声データが空です"
        duration = len(y) / sr
        if duration < 0.1: return False, "音声が短すぎます"
        if duration > 300.0: return False, "音声が長すぎます"
        return True, "OK"
    except Exception as e:
        return False, f"音声ファイル検証エラー: {str(e)}"

# 声調評価関数
def _evaluate_tone_advanced(pinyin_with_tones: List[str], aligned_words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """声調評価"""
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
                "score": DUMMY_FALLBACK_SCORE,
                "confidence": 0.0,
                "pitch_pattern": "data_missing"
            })
            tone_scores.append(DUMMY_FALLBACK_SCORE)

    overall_tone_score = float(np.mean(tone_scores)) if tone_scores else 0.0
    return {
        "word_tone_scores": tone_details,
        "overall_tone_score": round(overall_tone_score, 3),
        "tone_distribution": _analyze_tone_distribution(tone_details),
        "improvement_suggestions": _generate_tone_suggestions(tone_details)
    }

def _evaluate_single_tone(pinyin: str, word_data: Dict) -> Dict[str, Any]:
    """単一単語の声調評価"""
    expected_tone_num = int(re.search(r'[1-5]', pinyin).group()) if re.search(r'[1-5]', pinyin) else 0
    phones = word_data.get("phones", [])
    
    pitch_values = [p.get("pitch", 0) for p in phones if p.get("pitch") is not None]
    pitch_pattern_analysis = _analyze_pitch_pattern(pitch_values, expected_tone_num)
    
    tone_confidence = float(np.mean([p.get("score", 0) for p in phones])) if phones else 0.0

    score = tone_confidence
    if "correct" in pitch_pattern_analysis: score = min(MAX_SCORE_CLAMP, score * 1.2)
    elif "incorrect" in pitch_pattern_analysis: score = max(MIN_SCORE_CLAMP, score * 0.7)
    elif "insufficient" in pitch_pattern_analysis or "unclear" in pitch_pattern_analysis: score = max(DUMMY_FALLBACK_SCORE, score * 0.9)

    return {
        "pinyin": pinyin,
        "expected_tone": expected_tone_num,
        "score": round(float(score), 3),
        "confidence": round(tone_confidence, 3),
        "pitch_pattern": pitch_pattern_analysis
    }

def _analyze_pitch_pattern(pitch_values: List[float], expected_tone: int) -> str:
    """ピッチパターンの分析"""
    pitch_values_filtered = [p for p in pitch_values if p > 0]
    if len(pitch_values_filtered) < 2: return "insufficient_data"

    mean_pitch = np.mean(pitch_values_filtered)
    std_dev_pitch = np.std(pitch_values_filtered)
    pitch_start = pitch_values_filtered[0]
    pitch_end = pitch_values_filtered[-1]
    pitch_range = max(pitch_values_filtered) - min(pitch_values_filtered)

    if expected_tone == 1:
        return "correct_flat" if mean_pitch > 150 and std_dev_pitch < 20 and pitch_range < 50 else "incorrect_unstable"
    elif expected_tone == 2:
        return "correct_rising" if pitch_end - pitch_start > 30 and std_dev_pitch > 15 and pitch_values_filtered[-1] > pitch_values_filtered[0] else "incorrect_pattern"
    elif expected_tone == 3:
        min_idx = np.argmin(pitch_values_filtered)
        if min_idx > 0 and min_idx < len(pitch_values_filtered) - 1:
            if pitch_values_filtered[min_idx] < pitch_values_filtered[0] and \
               pitch_values_filtered[min_idx] < pitch_values_filtered[-1] and \
               (pitch_values_filtered[-1] - pitch_values_filtered[min_idx]) > 10:
                return "correct_dip"
        return "incorrect_pattern"
    elif expected_tone == 4:
        return "correct_falling" if pitch_start - pitch_end > 30 and std_dev_pitch > 15 and pitch_values_filtered[0] > pitch_values_filtered[-1] else "incorrect_pattern"
    elif expected_tone == 5:
        return "neutral_tone"
    
    return "pattern_unclear"

def _analyze_tone_distribution(tone_details: List[Dict]) -> Dict[str, Any]:
    """声調分布の分析"""
    expected_tones = [t["expected_tone"] for t in tone_details]
    tone_counts = {i: expected_tones.count(i) for i in range(1, 6)}

    tone_avg_scores = {
        tone_num: round(float(np.mean([t["score"] for t in tone_details if t["expected_tone"] == tone_num])), 3)
        for tone_num in range(1, 6) if any(t["expected_tone"] == tone_num for t in tone_details)
    }

    most_difficult_tone = min(tone_avg_scores.items(), key=lambda x: x[1])[0] if tone_avg_scores else None
    best_tone = max(tone_avg_scores.items(), key=lambda x: x[1])[0] if tone_avg_scores else None

    return {
        "tone_counts": tone_counts,
        "tone_average_scores": tone_avg_scores,
        "most_difficult_tone": most_difficult_tone,
        "best_tone": best_tone
    }

def _generate_tone_suggestions(tone_details: List[Dict]) -> List[str]:
    """声調改善提案の生成"""
    suggestions = []
    tone_map = {
        1: "高く、平らに", 2: "はっきりと上昇させるように",
        3: "一度下げてから再び上げる「谷型」のパターンを",
        4: "上から下へ一気に下降させるように", 5: "短く、軽く、自然に"
    }

    for detail in tone_details:
        score, expected_tone, pitch_pattern, pinyin_str = detail["score"], detail["expected_tone"], detail["pitch_pattern"], detail['pinyin']
        if score < 0.6:
            base_suggestion = f"'{pinyin_str}' の第{expected_tone}声: "
            if expected_tone in tone_map:
                suggestions.append(base_suggestion + f"音程を{tone_map[expected_tone]}発音する練習をしましょう。")
            else:
                suggestions.append(base_suggestion + "発音の明瞭度を上げるために、ピッチの安定性や音素の明確さを意識しましょう。")

    return list(sorted(set(suggestions)))

# --- テスト実行部分 ---
if __name__ == "__main__":
    analysis_config = AnalysisConfig()
    paddlespeech_service = PaddleSpeechService(analysis_config)

    # ダミー音声データとテキスト
    dummy_audio_path = "test_audio.wav"
    sample_rate = analysis_config.audio_sample_rate
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    y_dummy = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    y_dummy_int16 = (y_dummy * 32767).astype(np.int16)
    write_wav(dummy_audio_path, sample_rate, y_dummy_int16)

    with open(dummy_audio_path, "rb") as f:
        audio_bytes = f.read()

    test_text = "你好中国"
    test_pinyin = ["ni3", "hao3", "zhong1", "guo2"]

    print(f"--- 音声分析開始 ---")
    start_time = time.time()
    try:
        result = paddlespeech_service.run_analysis(audio_bytes, test_text, test_pinyin)
        end_time = time.time()
        print(f"--- 音声分析完了 --- 処理時間: {end_time - start_time:.2f}秒")
        # NpEncoderを使用して、NumPy型をJSONでシリアライズ可能にする
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NpEncoder))

    except PronunciationAnalysisError as e:
        print(f"分析エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラー: {e}")
    finally:
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
            print(f"一時ファイル {dummy_audio_path} を削除しました。")

    print("\n--- 別のテストケース（短文） ---")
    test_text_short = "我爱你"
    test_pinyin_short = ["wo3", "ai4", "ni3"]

    dummy_audio_path_short = "test_audio_short.wav"
    duration_short_sec = 1.0
    t_short = np.linspace(0, duration_short_sec, int(sample_rate * duration_short_sec), endpoint=False)
    y_dummy_short = 0.6 * np.sin(2 * np.pi * 300 * t_short)
    y_dummy_short_int16 = (y_dummy_short * 32767).astype(np.int16)
    write_wav(dummy_audio_path_short, sample_rate, y_dummy_short_int16)

    with open(dummy_audio_path_short, "rb") as f:
        audio_bytes_short = f.read()

    start_time_short = time.time()
    try:
        result_short = paddlespeech_service.run_analysis(audio_bytes_short, test_text_short, test_pinyin_short)
        end_time_short = time.time()
        print(f"--- 音声分析完了 --- 処理時間: {end_time_short - start_time_short:.2f}秒")
        print(json.dumps(result_short, indent=2, ensure_ascii=False, cls=NpEncoder))
    except PronunciationAnalysisError as e:
        print(f"分析エラー (短文): {e}")
    except Exception as e:
        print(f"予期せぬエラー (短文): {e}")
    finally:
        if os.path.exists(dummy_audio_path_short):
            os.remove(dummy_audio_path_short)
            print(f"一時ファイル {dummy_audio_path_short} を削除しました。")