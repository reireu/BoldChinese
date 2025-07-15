import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.io.wavfile import write as write_wav
from scipy.signal import find_peaks
import warnings
import tempfile
import shutil
import pypinyin
import os
import paddle
import librosa
import numpy as np
import json
import re
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path

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

@dataclass
class PaddleSpeechConfig:
    """PaddleSpeech設定"""
    asr_model: str = "conformer_wenetspeech"
    asr_lang: str = "zh"
    asr_sample_rate: int = 16000
    asr_config: Optional[str] = None
    asr_ckpt: Optional[str] = None
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

# ログ設定を強化
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        self._paddlespeech_available = False
        
        logger.info("PaddleSpeechService初期化開始")
        self._initialize_executors()
        logger.info(f"PaddleSpeechService初期化完了 - PaddleSpeech利用可能: {self._paddlespeech_available}")

    def _initialize_executors(self):
        """PaddleSpeech実行器の初期化"""
        try:
            logger.info("PaddleSpeechの可用性を確認中...")
            
            # Paddleのデバイス設定を確認
            try:
                device_info = paddle.get_device()
                logger.info(f"Paddle device: {device_info}")
            except Exception as e:
                logger.warning(f"Paddleデバイス情報取得エラー: {e}")
            
            # PaddleSpeechのインポートを試行
            logger.info("PaddleSpeech ASR and Text executors をインポート中...")
            from paddlespeech.cli.asr import ASRExecutor
            from paddlespeech.cli.text import TextExecutor
            
            logger.info("ASRExecutor初期化中...")
            self._asr_executor = ASRExecutor()
            logger.info("ASRExecutor初期化完了")
            
            logger.info("TextExecutor初期化中...")
            self._text_executor = TextExecutor()
            logger.info("TextExecutor初期化完了")
            
            self._paddlespeech_available = True
            logger.info("PaddleSpeech ASR and Text executors が正常に初期化されました")
            
        except ImportError as e:
            logger.error(f"PaddleSpeechインポートエラー: {e}")
            logger.warning("PaddleSpeech ASR or TextExecutor が見つかりません。PaddleSpeechが正しくインストールされているか確認してください。ダミー実装にフォールバックします。")
            self._asr_executor = None
            self._text_executor = None
            self._paddlespeech_available = False
            
        except Exception as e:
            logger.error(f"PaddleSpeech初期化で予期しないエラー: {e}", exc_info=True)
            logger.warning("PaddleSpeech初期化が予期しないエラーで失敗しました。ダミー実装にフォールバックします。")
            self._asr_executor = None
            self._text_executor = None
            self._paddlespeech_available = False

    def _extract_acoustic_features(self, audio_path: str) -> Dict[str, Union[float, List]]:
        """音響特徴量の抽出"""
        logger.info(f"音響特徴量抽出開始: {audio_path}")
        
        try:
            # 音声ファイルの存在確認
            if not os.path.exists(audio_path):
                raise AudioProcessingError(f"音声ファイルが存在しません: {audio_path}")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(audio_path)
            logger.info(f"音声ファイルサイズ: {file_size} bytes")
            
            if file_size == 0:
                raise AudioProcessingError(f"音声ファイルが空です: {audio_path}")
            
            # 音声データのロード
            logger.info(f"音声データロード中 (目標サンプルレート: {self.config.audio_sample_rate}Hz)")
            y, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)
            
            logger.info(f"音声データロード完了 - 実際のサンプルレート: {sr}Hz, データ長: {len(y)}")

            if len(y) == 0:
                raise AudioProcessingError(f"音声データが空です: {audio_path}")

            duration = len(y) / sr
            logger.info(f"音声継続時間: {duration:.3f}秒")
            
            if duration < 0.1:
                logger.warning(f"音声が短すぎます: {duration:.3f}秒")
            
            # ピッチの抽出
            logger.info("ピッチ抽出開始")
            try:
                # より安定したピッチ抽出のためにyinを使用
                f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
                pitch_values = [float(f) for f in f0 if f > 0]
                logger.info(f"ピッチ抽出完了 - 有効なピッチ値: {len(pitch_values)}")
            except Exception as e:
                logger.warning(f"YINピッチ抽出エラー、piptrackにフォールバック: {e}")
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.01)
                pitch_values = []
                for t_idx in range(pitches.shape[1]):
                    if magnitudes[:, t_idx].sum() > 0:
                        pitch_idx = np.argmax(magnitudes[:, t_idx])
                        if pitches[pitch_idx, t_idx] > 0:
                            pitch_values.append(float(pitches[pitch_idx, t_idx]))
                logger.info(f"piptrack ピッチ抽出完了 - 有効なピッチ値: {len(pitch_values)}")
            
            # RMS energy per frame
            logger.info("RMSエネルギー計算開始")
            rms_energy_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_energy_overall = float(np.mean(rms_energy_frames)) if len(rms_energy_frames) > 0 else 0.0
            logger.info(f"RMSエネルギー計算完了 - フレーム数: {len(rms_energy_frames)}, 平均エネルギー: {rms_energy_overall:.6f}")
            
            # その他の特徴量計算
            logger.info("その他の音響特徴量計算開始")
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            logger.info(f"スペクトル重心: {len(spectral_centroids)}フレーム")
            logger.info(f"スペクトルロールオフ: {len(spectral_rolloff)}フレーム")
            logger.info(f"MFCC: {mfccs.shape}")

            result = {
                "pitch_values": pitch_values,
                "mean_pitch": float(np.mean(pitch_values)) if pitch_values else 0.0,
                "pitch_std": float(np.std(pitch_values)) if pitch_values else 0.0,
                "duration": float(duration),
                "rms_energy_frames": rms_energy_frames.tolist(),
                "rms_energy_overall": rms_energy_overall,
                "spectral_centroids": spectral_centroids.tolist(),
                "spectral_rolloff": spectral_rolloff.tolist(),
                "mfccs": mfccs.tolist(),
                "sample_rate": sr,
                "audio_length": len(y)
            }
            
            logger.info("音響特徴量抽出完了")
            return result

        except Exception as e:
            logger.error(f"音響特徴量抽出エラー ({audio_path}): {e}", exc_info=True)
            raise AudioProcessingError(f"音響特徴量抽出に失敗しました: {e}")

    def _run_paddlespeech_asr(self, audio_path: str) -> Dict[str, Any]:
        """PaddleSpeech ASRを実行"""
        logger.info(f"ASR実行開始: {audio_path}")
        
        if not self._paddlespeech_available or self._asr_executor is None:
            logger.warning("PaddleSpeech ASRが利用できません。ダミー結果を返します。")
            return {"text": "识别结果", "confidence": DUMMY_ASR_CONFIDENCE, "segments": []}

        try:
            logger.info("PaddleSpeech ASR実行中...")
            logger.info(f"ASRパラメータ: model={self.ps_config.asr_model}, lang={self.ps_config.asr_lang}, sample_rate={self.ps_config.asr_sample_rate}")
            
            result = self._asr_executor(
                audio_file=audio_path,
                model=self.ps_config.asr_model,
                lang=self.ps_config.asr_lang,
                sample_rate=self.ps_config.asr_sample_rate,
                config=self.ps_config.asr_config,
                ckpt_path=self.ps_config.asr_ckpt,
                device=self.ps_config.device
            )
            
            logger.info(f"ASR実行完了。結果の型: {type(result)}")
            
            # PaddleSpeech CLIの戻り値の型を柔軟にハンドリング
            if isinstance(result, str):
                logger.info(f"ASR結果（文字列）: {result}")
                return {"text": result, "confidence": DUMMY_ASR_CONFIDENCE, "segments": []}
            elif isinstance(result, dict):
                logger.info(f"ASR結果（辞書）: {result}")
                return {
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", DUMMY_ASR_CONFIDENCE),
                    "segments": result.get("segments", [])
                }
            else:
                logger.info(f"ASR結果（オブジェクト）: {result}")
                return {
                    "text": getattr(result, 'text', ''),
                    "confidence": getattr(result, 'confidence', DUMMY_ASR_CONFIDENCE),
                    "segments": getattr(result, 'segments', [])
                }
                
        except Exception as e:
            logger.error(f"PaddleSpeech ASR実行エラー: {e}", exc_info=True)
            logger.warning("ASRエラーのためダミー結果を返します")
            return {"text": "识别结果", "confidence": DUMMY_ASR_CONFIDENCE, "segments": []}

    def _run_paddlespeech_alignment(self, audio_path: str, text: str, acoustic_features: Dict, asr_segments: List[Dict]) -> Dict[str, Any]:
        """強制アラインメントを実行"""
        logger.info(f"アラインメント実行開始: text='{text}', segments={len(asr_segments)}")
        return self._generate_enhanced_alignment(text, acoustic_features, asr_segments)

    def _text_to_phonemes(self, text: str) -> List[str]:
        """テキストを音素に変換"""
        logger.info(f"テキスト音素変換開始: '{text}'")
        
        if not self._paddlespeech_available or self._text_executor is None:
            logger.warning("PaddleSpeech TextExecutorが利用できません。pypinyinフォールバックを使用します。")
            return self._text_to_phonemes_fallback(text)

        try:
            logger.info("PaddleSpeech TextExecutorでピンイン変換実行中...")
            pinyin_str_from_ps = self._text_executor(text=text, lang="zh", to_pinyin=True)
            logger.info(f"TextExecutor結果: {pinyin_str_from_ps}")
            
            all_phonemes = []
            for pinyin_part in pinyin_str_from_ps.split():
                phonemes = self._pinyin_to_phonemes(pinyin_part)
                all_phonemes.extend(phonemes)
                logger.debug(f"ピンイン '{pinyin_part}' -> 音素 {phonemes}")
            
            logger.info(f"音素変換完了: {all_phonemes}")
            return all_phonemes
            
        except Exception as e:
            logger.error(f"PaddleSpeech Text音素変換エラー: {e}", exc_info=True)
            logger.warning("TextExecutorエラーのためpypinyinフォールバックを使用します")
            return self._text_to_phonemes_fallback(text)

    def _text_to_phonemes_fallback(self, text: str) -> List[str]:
        """テキストを音素に変換（pypinyinフォールバック）"""
        logger.info(f"pypinyinフォールバック音素変換: '{text}'")
        
        all_phonemes = []
        for char in text:
            pys = pypinyin.pinyin(char, style=pypinyin.Style.TONE2, heteronym=False)
            if pys and pys[0]:
                pinyin_str = pys[0][0]
                phonemes = self._pinyin_to_phonemes(pinyin_str)
                all_phonemes.extend(phonemes)
                logger.debug(f"文字 '{char}' -> ピンイン '{pinyin_str}' -> 音素 {phonemes}")
            else:
                all_phonemes.append("unknown")
                logger.debug(f"文字 '{char}' -> 音素 ['unknown']")
        
        logger.info(f"pypinyinフォールバック音素変換完了: {all_phonemes}")
        return all_phonemes

    def _calculate_phoneme_score(self, phoneme: str, energy_level: float, pitch_std_segment: float) -> float:
        """音素スコアの計算"""
        logger.debug(f"音素スコア計算: phoneme='{phoneme}', energy={energy_level:.6f}, pitch_std={pitch_std_segment:.3f}")
        
        base_score = DUMMY_ALIGN_SCORE

        # エネルギーレベルに基づいてスコアを調整
        if energy_level < 0.005:
            base_score = max(MIN_SCORE_CLAMP, base_score * 0.5)
            logger.debug(f"低エネルギー調整: {base_score}")
        elif energy_level < 0.02:
            base_score = max(MIN_SCORE_CLAMP, base_score * 0.8)
            logger.debug(f"やや低エネルギー調整: {base_score}")
        elif energy_level > 0.2:
            base_score = max(MIN_SCORE_CLAMP, base_score * 0.9)
            logger.debug(f"高エネルギー調整: {base_score}")

        # ピッチ標準偏差に基づく調整
        if phoneme not in INITIALS and phoneme not in FINALS:
            pass  # 声調番号や'unknown'はスキップ
        elif pitch_std_segment > 50:
            base_score = max(MIN_SCORE_CLAMP, base_score * 0.7)
            logger.debug(f"高ピッチ変動調整: {base_score}")
        elif pitch_std_segment < 5 and phoneme not in ['1', '5']:
            base_score = max(MIN_SCORE_CLAMP, base_score * 0.8)
            logger.debug(f"低ピッチ変動調整: {base_score}")

        difficulty = self._get_phoneme_difficulty(phoneme)
        base_score -= difficulty * 0.1
        logger.debug(f"難易度調整: difficulty={difficulty}, score={base_score}")

        # ランダム性を追加
        final_score = max(MIN_SCORE_CLAMP, min(MAX_SCORE_CLAMP, float(base_score + np.random.normal(0, 0.03))))
        logger.debug(f"最終音素スコア: {final_score}")
        
        return final_score

    def _get_pitch_at_time(self, start: float, end: float, acoustic_features: Dict) -> float:
        """指定時間でのピッチを取得"""
        pitch_values = acoustic_features.get("pitch_values", [])
        duration = acoustic_features.get("duration", 1.0)
        
        logger.debug(f"時間範囲のピッチ取得: {start:.3f}-{end:.3f}秒, 総ピッチ値数: {len(pitch_values)}")
        
        if not pitch_values or duration == 0:
            logger.debug("ピッチ値なしまたは継続時間ゼロ")
            return 0.0

        pitch_frame_rate = len(pitch_values) / duration
        start_idx = int(start * pitch_frame_rate)
        end_idx = int(end * pitch_frame_rate)
        
        logger.debug(f"ピッチインデックス範囲: {start_idx}-{end_idx}")

        segment_pitches = [p for p in pitch_values[start_idx:end_idx] if p > 0]
        result = float(np.mean(segment_pitches)) if segment_pitches else 0.0
        
        logger.debug(f"セグメントピッチ: {len(segment_pitches)}個, 平均: {result:.3f}")
        return result

    def _generate_enhanced_alignment(self, text: str, acoustic_features: Dict, asr_segments: List[Dict]) -> Dict[str, Any]:
        """強化されたアラインメントを生成"""
        logger.info(f"強化アラインメント生成開始: text='{text}', ASRセグメント数: {len(asr_segments)}")
        
        rms_energy_frames = np.array(acoustic_features.get("rms_energy_frames", []))
        duration = acoustic_features.get("duration", 1.0)
        sr = acoustic_features.get("sample_rate", self.config.audio_sample_rate)
        
        logger.info(f"音響特徴量: RMSフレーム数={len(rms_energy_frames)}, 継続時間={duration:.3f}秒, サンプルレート={sr}")
        
        if len(rms_energy_frames) == 0 or duration == 0:
            logger.error("RMSエネルギーデータなしまたは音声継続時間ゼロ")
            return {"phones": [], "pinyin_list_for_alignment": []}

        # RMSフレーム時間軸の計算
        rms_hop_length = 512
        frame_times = librosa.frames_to_time(np.arange(len(rms_energy_frames)), sr=sr, hop_length=rms_hop_length)
        
        logger.info(f"RMSフレーム時間: {len(frame_times)}フレーム")

        all_phones_aligned = []
        
        # テキストの各文字に対応するピンインリストを生成
        pinyin_list_for_alignment = []
        for char in list(text):
            pys = pypinyin.pinyin(char, style=pypinyin.Style.TONE2, heteronym=False)
            pinyin_for_char = pys[0][0] if pys and pys[0] else "unknown"
            pinyin_list_for_alignment.append(pinyin_for_char)
            logger.debug(f"文字 '{char}' -> ピンイン '{pinyin_for_char}'")

        # ASRセグメントの利用
        if asr_segments:
            logger.info("ASRセグメントを使用したアラインメント")
            for segment_idx, segment in enumerate(asr_segments):
                segment_text = segment.get("text", "")
                segment_start = segment.get("start", 0.0)
                segment_end = segment.get("end", duration)
                
                logger.info(f"セグメント{segment_idx}: '{segment_text}' ({segment_start:.3f}-{segment_end:.3f}秒)")
                
                if len(segment_text) == 0:
                    logger.warning("空のセグメントテキスト、スキップ")
                    continue

                # 文字ごとに時間を均等分割
                char_duration = (segment_end - segment_start) / len(segment_text)
                
                # 元のテキストでのセグメントの位置を探す
                text_start_idx = text.find(segment_text)
                if text_start_idx == -1:
                    logger.warning(f"セグメントテキスト '{segment_text}' が元のテキストに見つかりません")
                    continue

                for i, char_in_segment in enumerate(segment_text):
                    word_start = segment_start + i * char_duration
                    word_end = word_start + char_duration
                    
                    original_char_idx = text_start_idx + i
                    if original_char_idx >= len(pinyin_list_for_alignment):
                        logger.warning(f"文字インデックス範囲外: {original_char_idx}")
                        continue
                    
                    pinyin_for_char = pinyin_list_for_alignment[original_char_idx]
                    estimated_phonemes = self._pinyin_to_phonemes(pinyin_for_char)
                    
                    logger.debug(f"文字 '{char_in_segment}' -> ピンイン '{pinyin_for_char}' -> 音素 {estimated_phonemes}")
                    
                    if not estimated_phonemes:
                        continue

                    # 音素ごとに時間を均等分割
                    phoneme_duration = (word_end - word_start) / len(estimated_phonemes)
                    
                    for ph_idx, phoneme in enumerate(estimated_phonemes):
                        phone_start = word_start + (ph_idx * phoneme_duration)
                        phone_end = phone_start + phoneme_duration

                        # 音素区間のRMSエネルギーを計算
                        start_frame = int(phone_start * len(frame_times) / duration) if duration > 0 else 0
                        end_frame = int(phone_end * len(frame_times) / duration) if duration > 0 else len(rms_energy_frames)
                        
                        start_frame = max(0, min(start_frame, len(rms_energy_frames) - 1))
                        end_frame = max(start_frame + 1, min(end_frame, len(rms_energy_frames)))
                        
                        segment_rms = rms_energy_frames[start_frame:end_frame]
                        avg_rms = float(np.mean(segment_rms)) if len(segment_rms) > 0 else 1e-6
                        
                        # 音素区間のピッチを計算
                        mean_pitch = self._get_pitch_at_time(phone_start, phone_end, acoustic_features)
                        
                        # ピッチ標準偏差を計算
                        pitch_values_all = acoustic_features.get("pitch_values", [])
                        pitch_frame_rate = len(pitch_values_all) / duration if duration > 0 else 0
                        pitch_start_idx = int(phone_start * pitch_frame_rate)
                        pitch_end_idx = int(phone_end * pitch_frame_rate)
                        
                        pitch_start_idx = max(0, min(pitch_start_idx, len(pitch_values_all) - 1))
                        pitch_end_idx = max(pitch_start_idx + 1, min(pitch_end_idx, len(pitch_values_all)))
                        
                        segment_pitches = [p for p in pitch_values_all[pitch_start_idx:pitch_end_idx] if p > 0]
                        pitch_std = float(np.std(segment_pitches)) if len(segment_pitches) > 1 else 0.0

                        score = self._calculate_phoneme_score(phoneme, avg_rms, pitch_std)

                        phone_info = {
                            "phone": phoneme,
                            "start": phone_start,
                            "end": phone_end,
                            "score": score,
                            "pitch": mean_pitch,
                            "rms_energy": avg_rms
                        }
                        
                        all_phones_aligned.append(phone_info)
                        logger.debug(f"音素追加: {phone_info}")
        else:
            logger.warning("ASRセグメントなし、均等分割アラインメントを使用")
            # 均等分割フォールバック
            total_phonemes = sum(len(self._pinyin_to_phonemes(p)) for p in pinyin_list_for_alignment)
            if total_phonemes == 0:
                logger.error("推定音素数がゼロ")
                return {"phones": [], "pinyin_list_for_alignment": pinyin_list_for_alignment}
            
            avg_phoneme_duration = duration / total_phonemes
            logger.info(f"平均音素継続時間: {avg_phoneme_duration:.3f}秒")
            
            current_time = 0.0
            for char_idx, pinyin_for_char in enumerate(pinyin_list_for_alignment):
                estimated_phonemes = self._pinyin_to_phonemes(pinyin_for_char)
                
                if not estimated_phonemes:
                    continue
                
                for phoneme in estimated_phonemes:
                    phone_start = current_time
                    phone_end = current_time + avg_phoneme_duration
                    
                    # 音素区間のRMSエネルギーを計算
                    start_frame = int(phone_start * len(frame_times) / duration) if duration > 0 else 0
                    end_frame = int(phone_end * len(frame_times) / duration) if duration > 0 else len(rms_energy_frames)
                    
                    start_frame = max(0, min(start_frame, len(rms_energy_frames) - 1))
                    end_frame = max(start_frame + 1, min(end_frame, len(rms_energy_frames)))
                    
                    segment_rms = rms_energy_frames[start_frame:end_frame]
                    avg_rms = float(np.mean(segment_rms)) if len(segment_rms) > 0 else 1e-6
                    
                    # 音素区間のピッチを計算
                    mean_pitch = self._get_pitch_at_time(phone_start, phone_end, acoustic_features)
                    
                    # ピッチ標準偏差を計算
                    pitch_values_all = acoustic_features.get("pitch_values", [])
                    pitch_frame_rate = len(pitch_values_all) / duration if duration > 0 else 0
                    pitch_start_idx = int(phone_start * pitch_frame_rate)
                    pitch_end_idx = int(phone_end * pitch_frame_rate)
                    
                    pitch_start_idx = max(0, min(pitch_start_idx, len(pitch_values_all) - 1))
                    pitch_end_idx = max(pitch_start_idx + 1, min(pitch_end_idx, len(pitch_values_all)))
                    
                    segment_pitches = [p for p in pitch_values_all[pitch_start_idx:pitch_end_idx] if p > 0]
                    pitch_std = float(np.std(segment_pitches)) if len(segment_pitches) > 1 else 0.0
                    
                    score = self._calculate_phoneme_score(phoneme, avg_rms, pitch_std)
                    
                    phone_info = {
                        "phone": phoneme,
                        "start": phone_start,
                        "end": phone_end,
                        "score": score,
                        "pitch": mean_pitch,
                        "rms_energy": avg_rms
                    }
                    
                    all_phones_aligned.append(phone_info)
                    logger.debug(f"音素追加（均等分割）: {phone_info}")
                    
                    current_time = phone_end
        
        logger.info(f"アラインメント完了: {len(all_phones_aligned)}個の音素")
        return {"phones": all_phones_aligned, "pinyin_list_for_alignment": pinyin_list_for_alignment}

    def _pinyin_to_phonemes(self, pinyin: str) -> List[str]:
        """ピンインを音素に分解"""
        if not pinyin or pinyin == "unknown":
            return ["unknown"]
        
        # 声調番号を分離
        tone_match = re.search(r'[1-5]$', pinyin)
        tone = tone_match.group() if tone_match else ""
        pinyin_no_tone = pinyin.rstrip('12345')
        
        logger.debug(f"ピンイン分解: '{pinyin}' -> base='{pinyin_no_tone}', tone='{tone}'")
        
        # 声母（初期子音）の抽出
        initial = ""
        for init in INITIALS:
            if pinyin_no_tone.startswith(init):
                initial = init
                break
        
        # 韻母（母音＋最終子音）の抽出
        final = pinyin_no_tone[len(initial):] if initial else pinyin_no_tone
        
        # 最終的な音素リスト
        phonemes = []
        if initial:
            phonemes.append(initial)
        
        if final:
            # 韻母をさらに分解
            for fin in FINALS:
                if final == fin:
                    phonemes.append(fin)
                    break
            else:
                if final:  # 既知の韻母でない場合もそのまま追加
                    phonemes.append(final)
        
        if tone:
            phonemes.append(tone)
        
        logger.debug(f"音素分解結果: {phonemes}")
        return phonemes

    def _get_phoneme_difficulty(self, phoneme: str) -> float:
        """音素の難易度を返す（0.0-1.0）"""
        # 難易度定義
        difficult_initials = ['zh', 'ch', 'sh', 'r', 'z', 'c', 's']
        difficult_finals = ['iang', 'iong', 'uang', 'ueng', 'ue', 'v']
        
        if phoneme in difficult_initials:
            return 0.8
        elif phoneme in difficult_finals:
            return 0.7
        elif phoneme in ['j', 'q', 'x']:
            return 0.6
        elif phoneme in ['3', '4']:  # 3声、4声
            return 0.5
        elif phoneme in INITIALS or phoneme in FINALS:
            return 0.3
        else:
            return 0.2

    def analyze_pronunciation(self, audio_path: str, target_text: str) -> Dict[str, Any]:
        """発音分析のメイン処理"""
        logger.info(f"発音分析開始: audio='{audio_path}', text='{target_text}'")
        
        try:
            # 入力検証
            self._validate_inputs(audio_path, target_text)
            
            # 1. 音響特徴量の抽出
            logger.info("ステップ1: 音響特徴量抽出")
            acoustic_features = self._extract_acoustic_features(audio_path)
            
            # 2. ASR実行
            logger.info("ステップ2: ASR実行")
            asr_result = self._run_paddlespeech_asr(audio_path)
            
            # 3. 強制アラインメント
            logger.info("ステップ3: 強制アラインメント")
            alignment_result = self._run_paddlespeech_alignment(
                audio_path, target_text, acoustic_features, asr_result.get("segments", [])
            )
            
            # 4. 音素別評価
            logger.info("ステップ4: 音素別評価")
            phoneme_scores = self._evaluate_phonemes(alignment_result["phones"])
            
            # 5. 全体スコア計算
            logger.info("ステップ5: 全体スコア計算")
            overall_score = self._calculate_overall_score(phoneme_scores, acoustic_features)
            
            # 6. 結果の統合
            result = {
                "overall_score": overall_score,
                "asr_result": asr_result,
                "acoustic_features": acoustic_features,
                "alignment": alignment_result,
                "phoneme_scores": phoneme_scores,
                "target_text": target_text,
                "analysis_timestamp": time.time()
            }
            
            logger.info(f"発音分析完了: 全体スコア={overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"発音分析エラー: {e}", exc_info=True)
            raise PronunciationAnalysisError(f"発音分析に失敗しました: {e}")

    def _validate_inputs(self, audio_path: str, target_text: str):
        """入力検証"""
        if not audio_path:
            raise ValidationError("音声ファイルパスが空です")
        
        if not os.path.exists(audio_path):
            raise ValidationError(f"音声ファイルが存在しません: {audio_path}")
        
        if not target_text or not target_text.strip():
            raise ValidationError("対象テキストが空です")
        
        if len(target_text) > self.config.max_text_length:
            raise ValidationError(f"テキストが長すぎます（最大{self.config.max_text_length}文字）")
        
        if len(target_text) < self.config.min_text_length:
            raise ValidationError(f"テキストが短すぎます（最小{self.config.min_text_length}文字）")

    def _evaluate_phonemes(self, phones: List[Dict]) -> List[Dict]:
        """音素の評価"""
        logger.info(f"音素評価開始: {len(phones)}個の音素")
        
        phoneme_scores = []
        for phone in phones:
            score_info = {
                "phone": phone["phone"],
                "start": phone["start"],
                "end": phone["end"],
                "score": phone["score"],
                "pitch": phone["pitch"],
                "rms_energy": phone["rms_energy"],
                "difficulty": self._get_phoneme_difficulty(phone["phone"])
            }
            phoneme_scores.append(score_info)
        
        logger.info(f"音素評価完了: 平均スコア={np.mean([p['score'] for p in phoneme_scores]):.3f}")
        return phoneme_scores

    def _calculate_overall_score(self, phoneme_scores: List[Dict], acoustic_features: Dict) -> float:
        """全体スコアの計算"""
        if not phoneme_scores:
            logger.warning("音素スコアがないため、フォールバックスコアを返します")
            return DUMMY_FALLBACK_SCORE
        
        # 音素スコアの平均
        avg_phoneme_score = np.mean([p["score"] for p in phoneme_scores])
        
        # 音響特徴量に基づく調整
        pitch_consistency = self._calculate_pitch_consistency(acoustic_features)
        energy_consistency = self._calculate_energy_consistency(acoustic_features)
        
        # 重み付け計算
        overall_score = (
            avg_phoneme_score * 0.6 +
            pitch_consistency * 0.2 +
            energy_consistency * 0.2
        )
        
        # スコアのクランプ
        final_score = max(MIN_SCORE_CLAMP, min(MAX_SCORE_CLAMP, overall_score))
        
        logger.info(f"全体スコア計算: 音素平均={avg_phoneme_score:.3f}, "
                   f"ピッチ一貫性={pitch_consistency:.3f}, "
                   f"エネルギー一貫性={energy_consistency:.3f}, "
                   f"最終スコア={final_score:.3f}")
        
        return float(final_score)

    def _calculate_pitch_consistency(self, acoustic_features: Dict) -> float:
        """ピッチ一貫性の計算"""
        pitch_values = acoustic_features.get("pitch_values", [])
        if len(pitch_values) < 2:
            return 0.5
        
        # ピッチの変動係数（標準偏差/平均）
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        if pitch_mean == 0:
            return 0.5
        
        cv = pitch_std / pitch_mean
        
        # 変動係数を0-1のスコアに変換（低い変動 = 高いスコア）
        consistency_score = max(0.0, min(1.0, 1.0 - cv))
        
        logger.debug(f"ピッチ一貫性: CV={cv:.3f}, スコア={consistency_score:.3f}")
        return consistency_score

    def _calculate_energy_consistency(self, acoustic_features: Dict) -> float:
        """エネルギー一貫性の計算"""
        rms_energy_frames = acoustic_features.get("rms_energy_frames", [])
        if len(rms_energy_frames) < 2:
            return 0.5
        
        # エネルギーの変動係数
        energy_mean = np.mean(rms_energy_frames)
        energy_std = np.std(rms_energy_frames)
        
        if energy_mean == 0:
            return 0.5
        
        cv = energy_std / energy_mean
        
        # 変動係数を0-1のスコアに変換
        consistency_score = max(0.0, min(1.0, 1.0 - cv * 0.5))
        
        logger.debug(f"エネルギー一貫性: CV={cv:.3f}, スコア={consistency_score:.3f}")
        return consistency_score

    def create_analysis_report(self, analysis_result: Dict) -> Dict[str, Any]:
        """分析結果からレポートを作成"""
        logger.info("分析レポート作成開始")
        
        overall_score = analysis_result["overall_score"]
        phoneme_scores = analysis_result["phoneme_scores"]
        acoustic_features = analysis_result["acoustic_features"]
        
        # 音素別の詳細分析
        phoneme_analysis = []
        for p_score in phoneme_scores:
            phoneme_analysis.append({
                "phoneme": p_score["phone"],
                "score": p_score["score"],
                "difficulty": p_score["difficulty"],
                "timing": f"{p_score['start']:.3f}-{p_score['end']:.3f}s",
                "pitch": p_score["pitch"],
                "energy": p_score["rms_energy"]
            })
        
        # 改善提案
        suggestions = self._generate_improvement_suggestions(phoneme_scores, acoustic_features)
        
        # 統計情報
        stats = {
            "total_phonemes": len(phoneme_scores),
            "average_score": float(np.mean([p["score"] for p in phoneme_scores])) if phoneme_scores else 0.0,
            "difficult_phonemes": len([p for p in phoneme_scores if p["difficulty"] > 0.6]),
            "duration": acoustic_features.get("duration", 0.0),
            "pitch_range": {
                "min": float(np.min(acoustic_features.get("pitch_values", [0]))) if acoustic_features.get("pitch_values") else 0.0,
                "max": float(np.max(acoustic_features.get("pitch_values", [0]))) if acoustic_features.get("pitch_values") else 0.0,
                "mean": acoustic_features.get("mean_pitch", 0.0)
            }
        }
        
        report = {
            "overall_score": overall_score,
            "score_level": self._get_score_level(overall_score),
            "phoneme_analysis": phoneme_analysis,
            "statistics": stats,
            "suggestions": suggestions,
            "timestamp": analysis_result.get("analysis_timestamp", time.time())
        }
        
        logger.info("分析レポート作成完了")
        return report

    def _generate_improvement_suggestions(self, phoneme_scores: List[Dict], acoustic_features: Dict) -> List[str]:
        """改善提案の生成"""
        suggestions = []
        
        if not phoneme_scores:
            return ["音素データが不足しています。より明確に発音してください。"]
        
        # 低スコア音素の分析
        low_score_phonemes = [p for p in phoneme_scores if p["score"] < 0.6]
        if low_score_phonemes:
            difficult_phonemes = [p["phone"] for p in low_score_phonemes]
            suggestions.append(f"以下の音素の発音を練習してください: {', '.join(set(difficult_phonemes))}")
        
        # ピッチ分析
        pitch_values = acoustic_features.get("pitch_values", [])
        if pitch_values:
            pitch_std = np.std(pitch_values)
            if pitch_std > 50:
                suggestions.append("ピッチの変動が大きすぎます。より安定した発音を心がけてください。")
            elif pitch_std < 5:
                suggestions.append("ピッチの変動が少なすぎます。声調をより明確に表現してください。")
        
        # エネルギー分析
        rms_energy_frames = acoustic_features.get("rms_energy_frames", [])
        if rms_energy_frames:
            energy_mean = np.mean(rms_energy_frames)
            if energy_mean < 0.01:
                suggestions.append("音量が小さすぎます。より大きな声で発音してください。")
            elif energy_mean > 0.3:
                suggestions.append("音量が大きすぎます。適度な音量で発音してください。")
        
        # 特定音素の提案
        zh_ch_sh_scores = [p for p in phoneme_scores if p["phone"] in ["zh", "ch", "sh"]]
        if zh_ch_sh_scores and np.mean([p["score"] for p in zh_ch_sh_scores]) < 0.5:
            suggestions.append("舌面音（zh, ch, sh）の発音を重点的に練習してください。")
        
        if not suggestions:
            suggestions.append("とても良い発音です！継続して練習してください。")
        
        return suggestions

    def _get_score_level(self, score: float) -> str:
        """スコアレベルの判定"""
        if score >= 0.9:
            return "優秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "普通"
        elif score >= 0.6:
            return "要改善"
        else:
            return "要練習"

    def save_analysis_result(self, analysis_result: Dict, output_path: str):
        """分析結果の保存"""
        logger.info(f"分析結果保存: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2, cls=NpEncoder)
            logger.info("分析結果保存完了")
            
        except Exception as e:
            logger.error(f"分析結果保存エラー: {e}")
            raise PronunciationAnalysisError(f"分析結果の保存に失敗しました: {e}")

    def create_visualization(self, analysis_result: Dict, output_path: str):
        """分析結果の可視化"""
        logger.info(f"可視化作成: {output_path}")
        
        try:
            plt.figure(figsize=(15, 10))
            
            # 音素スコアのプロット
            plt.subplot(2, 2, 1)
            phoneme_scores = analysis_result["phoneme_scores"]
            if phoneme_scores:
                phones = [p["phone"] for p in phoneme_scores]
                scores = [p["score"] for p in phoneme_scores]
                plt.bar(range(len(phones)), scores)
                plt.xticks(range(len(phones)), phones, rotation=45)
                plt.ylabel("スコア")
                plt.title("音素別スコア")
                plt.ylim(0, 1)
            
            # ピッチ変化のプロット
            plt.subplot(2, 2, 2)
            acoustic_features = analysis_result["acoustic_features"]
            pitch_values = acoustic_features.get("pitch_values", [])
            if pitch_values:
                duration = acoustic_features.get("duration", 1.0)
                time_axis = np.linspace(0, duration, len(pitch_values))
                plt.plot(time_axis, pitch_values)
                plt.xlabel("時間 (秒)")
                plt.ylabel("ピッチ (Hz)")
                plt.title("ピッチ変化")
            
            # エネルギー変化のプロット
            plt.subplot(2, 2, 3)
            rms_energy_frames = acoustic_features.get("rms_energy_frames", [])
            if rms_energy_frames:
                duration = acoustic_features.get("duration", 1.0)
                time_axis = np.linspace(0, duration, len(rms_energy_frames))
                plt.plot(time_axis, rms_energy_frames)
                plt.xlabel("時間 (秒)")
                plt.ylabel("RMSエネルギー")
                plt.title("エネルギー変化")
            
            # 全体スコア表示
            plt.subplot(2, 2, 4)
            overall_score = analysis_result["overall_score"]
            plt.bar(["全体スコア"], [overall_score], color='green' if overall_score >= 0.7 else 'orange' if overall_score >= 0.5 else 'red')
            plt.ylim(0, 1)
            plt.title(f"全体スコア: {overall_score:.3f}")
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("可視化作成完了")
            
        except Exception as e:
            logger.error(f"可視化作成エラー: {e}")
            raise PronunciationAnalysisError(f"可視化の作成に失敗しました: {e}")


# 使用例
def main():
    """メイン関数"""
    # 設定の作成
    config = AnalysisConfig(
        max_text_length=200,
        audio_sample_rate=16000,
        tone_weight=0.4,
        pronunciation_weight=0.6
    )
    
    # サービスの初期化
    service = PaddleSpeechService(config)
    
    # 分析の実行例
    try:
        # Step 1: Create a dummy audio file for demonstration
        # This will create a simple sine wave as a placeholder.
        sample_rate = config.audio_sample_rate # 16000 Hz
        duration = 2.0  # seconds
        frequency = 440.0  # Hz (A4 note)
        
        t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        amplitude = 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        
        # Normalize to 16-bit PCM range
        data = data * (2**15 - 1) / np.max(np.abs(data))
        data = data.astype(np.int16)

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "sample_audio.wav")
        write_wav(audio_path, sample_rate, data)
        logger.info(f"ダミー音声ファイルを生成しました: {audio_path}")

        target_text = "你好世界" # Your target Chinese text
        
        # 発音分析
        analysis_result = service.analyze_pronunciation(audio_path, target_text)
        
        # レポート作成
        report = service.create_analysis_report(analysis_result)
        
        # 結果の保存
        output_json_path = "analysis_result.json"
        service.save_analysis_result(analysis_result, output_json_path)
        
        # 可視化
        output_png_path = "analysis_visualization.png"
        service.create_visualization(analysis_result, output_png_path)
        
        print(f"分析完了: 全体スコア = {analysis_result['overall_score']:.3f}")
        print(f"レベル: {report['score_level']}")
        print("改善提案:")
        for suggestion in report['suggestions']:
            print(f"  - {suggestion}")
        
        print(f"\n詳細な結果は '{output_json_path}' に保存されました。")
        print(f"可視化は '{output_png_path}' に保存されました。")
            
    except Exception as e:
        logger.error(f"分析実行エラー: {e}", exc_info=True)
        print(f"エラーが発生しました: {e}")
    finally:
        # Clean up the temporary directory
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"一時ディレクトリを削除しました: {temp_dir}")

if __name__ == "__main__":
    main()