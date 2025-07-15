import argparse
from dataclasses import dataclass
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import logging
import librosa
import soundfile as sf
from datetime import datetime

# PaddleSpeech関連のインポート
try:
    from paddlespeech.cli.asr import ASRExecutor
    from paddlespeech.cli.text import TextExecutor
    # from paddlespeech.cli.vector import VectorExecutor # vectorは今回のアラインメントに直接関係ないので残すかはお任せ
    import paddle
    PADDLESPEECH_AVAILABLE = True
except ImportError:
    PADDLESPEECH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PaddleSpeechConfig:
    # ASR設定
    asr_model: str = "conformer_wenetspeech-zh-16k"  # 中国語モデル
    asr_lang: str = "zh"
    asr_sample_rate: int = 16000
    
    # 音声処理設定
    target_sample_rate: int = 16000
    audio_channels: int = 1
    
    # デバイス設定
    device: str = "cpu"  # or "gpu"
    
    # 音声品質の閾値
    min_audio_duration: float = 0.5  # 最小音声長（秒）
    max_audio_duration: float = 60.0  # 最大音声長（秒）
    silence_threshold: float = 0.01  # 無音判定の閾値

# PaddleSpeechAlignmentError は、AlignExecutor がない場合、
# その機能の欠如によるエラーを示すカスタム例外として残しておくことは可能
class PaddleSpeechAlignmentError(Exception):
    pass

class PaddleSpeechService:
    def __init__(self, config: PaddleSpeechConfig):
        self.config = config
        self.asr_executor = None
        self.text_executor = None
        # self.align_executor は使用しないため、初期化も行わない
        
    def check_paddlespeech_environment(self) -> Tuple[bool, List[str]]:
        """PaddleSpeech環境をチェック"""
        logger.info("PaddleSpeech環境をチェック中...")
        errors = []
        
        try:
            if not PADDLESPEECH_AVAILABLE:
                errors.append("PaddleSpeechのコアコンポーネント（ASRExecutor, TextExecutorなど）がインストールされていません。pip install paddlespeech でインストールしてください。")
                return False, errors
                
            # Paddleの動作確認
            try:
                paddle_version = paddle.__version__
                logger.info(f"PaddlePaddle version: {paddle_version}")
            except Exception as e:
                errors.append(f"PaddlePaddle初期化エラー: {e}")
                
            # デバイス確認
            if self.config.device == "gpu":
                if not paddle.is_compiled_with_cuda():
                    logger.warning("CUDA対応PaddlePaddleではありません。CPUを使用します。")
                    self.config.device = "cpu"
                else:
                    logger.info("GPU使用可能")
            
            # ASR Executorの初期化テスト
            try:
                self.asr_executor = ASRExecutor()
                logger.info("ASRExecutor初期化成功")
            except Exception as e:
                errors.append(f"ASRExecutor初期化エラー: {e}")
                
            # Text Executorの初期化テスト
            try:
                self.text_executor = TextExecutor()
                logger.info("TextExecutor初期化成功")
            except Exception as e:
                errors.append(f"TextExecutor初期化エラー: {e}")
                
            if not errors:
                logger.info("PaddleSpeech環境チェック完了")
                return True, []
            else:
                logger.error(f"PaddleSpeech環境チェックで{len(errors)}個のエラーが発生")
                return False, errors
                
        except Exception as e:
            error_msg = f"PaddleSpeech環境チェック中の予期しないエラー: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return False, errors
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """音声ファイルの妥当性をチェック"""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
        
        if path.stat().st_size == 0:
            raise ValueError(f"音声ファイルが空です: {audio_path}")
            
        # librosaで音声ファイルを読み込んで検証
        try:
            y, sr = librosa.load(audio_path, sr=None) # 元のサンプルレートで読み込み
            duration = len(y) / sr
            
            if duration < self.config.min_audio_duration:
                raise ValueError(f"音声が短すぎます: {duration:.2f}秒 (最小: {self.config.min_audio_duration}秒)")
                
            if duration > self.config.max_audio_duration:
                raise ValueError(f"音声が長すぎます: {duration:.2f}秒 (最大: {self.config.max_audio_duration}秒)")
                
            # 無音チェック (前処理前に一度行うことで、無音のファイルを弾く)
            # 全体的なRMSエネルギーが非常に低い場合を無音とみなす
            rms_val = np.sqrt(np.mean(y**2))
            if rms_val < self.config.silence_threshold:
                raise ValueError("音声が無音または音量が小さすぎます")
                
            logger.info(f"音声ファイル検証完了: {audio_path} (長さ: {duration:.2f}秒, サンプルレート: {sr}Hz)")
            return True
            
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)): # 自前でraiseした例外はそのまま
                raise
            else: # librosaの読み込みエラーなど
                raise ValueError(f"音声ファイルの読み込みエラーまたは破損: {e}")
    
    def preprocess_audio(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """音声ファイルを前処理（リサンプリング、モノラル変換、正規化、無音トリミング）"""
        try:
            # librosaで音声を読み込み（指定されたターゲットサンプルレートでリサンプリングも行う）
            y, sr = librosa.load(input_path, sr=self.config.target_sample_rate)
            
            # モノラルに変換（既にモノラルの場合は何もしない）
            if y.ndim > 1: # y.shapeでなく.ndimを使う
                y = librosa.to_mono(y)
                
            # 音量正規化 (ピーク値で正規化)
            y = librosa.util.normalize(y)
            
            # 無音部分をトリミング (top_db: デシベル単位の閾値。小さいほど積極的)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25) # 20dBは厳しすぎる場合があるので25dBに調整
            
            # トリミング後に音声が空になっていないかチェック
            if len(y_trimmed) == 0:
                return False, "音声から有効な部分が検出されませんでした（無音部分が多すぎる可能性があります）"

            # soundfileで保存
            sf.write(output_path, y_trimmed, self.config.target_sample_rate, subtype='PCM_16') # PCM_16で保存形式を明確に
            
            output_path_obj = Path(output_path)
            if not output_path_obj.exists() or output_path_obj.stat().st_size == 0:
                return False, f"前処理後のファイルが作成されませんでした、または空です: {output_path}"
                
            logger.info(f"音声前処理完了: {output_path} (長さ: {len(y_trimmed)/self.config.target_sample_rate:.2f}秒)")
            return True, ""
            
        except Exception as e:
            error_msg = f"音声前処理エラー: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def perform_asr(self, audio_path: str) -> Tuple[str, float, List[Dict]]:
        """音声認識を実行し、認識テキスト、信頼度、セグメント情報（もしあれば）を返す"""
        try:
            if self.asr_executor is None:
                # 環境チェックで初期化が失敗していたら、ここで再度試みる
                _env_ok, _ = self.check_paddlespeech_environment()
                if not _env_ok or self.asr_executor is None:
                    raise RuntimeError("ASRExecutorが利用できません。")

            # ASR実行
            result = self.asr_executor(
                audio_file=audio_path,
                model=self.config.asr_model,
                lang=self.config.asr_lang,
                sample_rate=self.config.asr_sample_rate,
                device=self.config.device
            )
            
            recognized_text = ""
            confidence = 0.0 # デフォルト信頼度
            segments = []

            # 結果の解析 (PaddleSpeech CLIの出力形式はバージョンやモデルにより異なるため、柔軟に対応)
            if isinstance(result, str):
                recognized_text = result
                confidence = 0.8 # 文字列の場合はデフォルト信頼度
            elif isinstance(result, dict):
                recognized_text = result.get('text', '')
                confidence = result.get('confidence', 0.8)
                segments = result.get('segments', []) # セグメント情報があれば取得
            else: # オブジェクト形式の場合 (例: CLIの出力オブジェクト)
                recognized_text = getattr(result, 'text', '')
                confidence = getattr(result, 'confidence', 0.8)
                segments = getattr(result, 'segments', [])

            # 認識テキストが空の場合の信頼度調整
            if not recognized_text.strip():
                confidence = 0.0
                logger.warning("ASR結果が空テキストでした。信頼度を0に設定。")

            logger.info(f"ASR結果: '{recognized_text}' (信頼度: {confidence:.3f})")
            return recognized_text, confidence, segments
            
        except Exception as e:
            error_msg = f"音声認識エラー: {e}"
            logger.error(error_msg, exc_info=True)
            # ASRエラー時は、認識テキストを空、信頼度を低くしてフォールバック
            return "", 0.1, [] # 0.1は、全く認識できなかった場合でも0より少し高い値
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算（Levenshtein距離ベース）"""
        try:
            if not text1 or not text2:
                return 0.0
                
            # 空白を除去して比較
            text1_clean = text1.replace(' ', '').replace('\n', '').replace('\t', '')
            text2_clean = text2.replace(' ', '').replace('\n', '').replace('\t', '')
            
            if not text1_clean or not text2_clean:
                return 0.0
            
            # Levenshtein距離を計算する関数 (再掲)
            def levenshtein_distance(s1, s2):
                if len(s1) > len(s2):
                    s1, s2 = s2, s1
                distances = range(len(s1) + 1)
                for i2, c2 in enumerate(s2):
                    distances_ = [i2 + 1]
                    for i1, c1 in enumerate(s1):
                        if c1 == c2:
                            distances_.append(distances[i1])
                        else:
                            distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                    distances = distances_
                return distances[-1]
            
            distance = levenshtein_distance(text1_clean, text2_clean)
            max_len = max(len(text1_clean), len(text2_clean))
            
            if max_len == 0:
                return 1.0 # 両方空文字列なら一致とみなす
                
            similarity = 1 - (distance / max_len)
            return max(0.0, similarity) # 0未満にならないように
            
        except Exception as e:
            logger.error(f"テキスト類似度計算エラー: {e}", exc_info=True)
            return 0.0
    
    def analyze_pronunciation_quality(self, audio_path: str, recognized_text: str, 
                                    original_text: str, asr_confidence: float, 
                                    asr_segments: List[Dict]) -> Dict[str, Any]:
        """発音品質を分析（AlignExecutorなしバージョン）"""
        try:
            y, sr = librosa.load(audio_path, sr=self.config.target_sample_rate)
            duration = len(y) / sr
            
            # 音声特徴量の抽出
            rms_energy_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            mean_energy = np.mean(rms_energy_frames)
            
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=400) # ピッチ検出範囲を調整
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            # ここで、AlignExecutorがないため、詳細なアラインメントに基づく声調・音素評価は行いません。
            # 代わりに、全体的な情報からスコアを算出します。
            
            # 発音品質スコアの計算 (基本スコア)
            pronunciation_score = 60.0 # 中央値から開始

            # ASR信頼度とテキスト一致度を主要因とする
            # 信頼度が高いほど、発音も良い可能性が高い
            pronunciation_score += asr_confidence * 30 # 0.0-1.0 -> 0-30点
            
            # 元のテキストと認識テキストの類似度
            text_similarity = self.calculate_text_similarity(recognized_text, original_text)
            pronunciation_score += text_similarity * 30 # 0.0-1.0 -> 0-30点
            
            # 音声の品質要素を加味
            # 音量（平均エネルギー）
            if mean_energy > 0.02 and mean_energy < 0.2: # 適切な音量範囲
                pronunciation_score += 5
            
            # 音声の長さ (自然な発話長)
            if self.config.min_audio_duration <= duration <= self.config.max_audio_duration:
                 pronunciation_score += 5
            elif duration < self.config.min_audio_duration * 0.5 or duration > self.config.max_audio_duration * 1.5:
                 pronunciation_score -= 10 # 極端な場合は減点

            # ピッチの変動（声調の有無、不自然さの指標）
            if pitch_values:
                pitch_std = np.std(pitch_values)
                # 中国語の声調は変化を伴うため、ある程度のピッチ変動は必要
                if 15 < pitch_std < 120: # 適度な声調変化の範囲（調整推奨）
                    pronunciation_score += 5
                elif pitch_std < 5: # 平坦すぎる（第一声以外では問題）
                    pronunciation_score -= 5
                elif pitch_std > 150: # 変動しすぎ（不自然）
                    pronunciation_score -= 5
            else: # ピッチ検出できなかった場合
                pronunciation_score -= 10 # 検出できないのは問題

            # スコアのクランプ
            pronunciation_score = min(100, max(0, pronunciation_score))

            # 各要素のスコア (簡易的なもの、詳細なアラインメントなしでは精緻な評価は困難)
            # ASR結果とテキスト類似度を強く反映させる
            intonation_score = pronunciation_score * (0.5 + 0.5 * text_similarity)
            rhythm_score = pronunciation_score * (0.5 + 0.5 * text_similarity)
            
            # ASRのsegmentsがある場合、簡易的なリズム評価の補助にする
            if asr_segments:
                # 実際の文字数とASRセグメントの文字数の比率
                if len(original_text) > 0:
                    asr_char_count = sum(len(seg.get('text', '')) for seg in asr_segments)
                    char_coverage = asr_char_count / len(original_text)
                    rhythm_score = rhythm_score * (0.8 + 0.2 * min(1.0, char_coverage)) # 認識された部分が多いほどリズムが良いと仮定

            return {
                "overall_pronunciation_score": round(pronunciation_score, 2), # 総合発音スコア
                "intonation_score": round(min(100, max(0, intonation_score)), 2), # イントネーションスコア
                "rhythm_score": round(min(100, max(0, rhythm_score)), 2),       # リズムスコア
                "audio_features": {
                    "duration": round(duration, 2),
                    "mean_energy": round(mean_energy, 4),
                    "pitch_variation": round(np.std(pitch_values) if pitch_values else 0, 2),
                    "pitch_values_count": len(pitch_values)
                },
                "text_analysis": {
                    "text_similarity": round(text_similarity, 3),
                    "asr_confidence": round(asr_confidence, 3)
                },
                "asr_segments_available": bool(asr_segments) # セグメント情報が利用されたか
            }
            
        except Exception as e:
            logger.error(f"発音品質分析エラー: {e}", exc_info=True)
            # エラー時も最小限の情報を返す
            return {
                "overall_pronunciation_score": 0.0,
                "intonation_score": 0.0,
                "rhythm_score": 0.0,
                "audio_features": {
                    "duration": 0.0,
                    "mean_energy": 0.0,
                    "pitch_variation": 0.0,
                    "pitch_values_count": 0
                },
                "text_analysis": {
                    "text_similarity": 0.0,
                    "asr_confidence": asr_confidence
                },
                "asr_segments_available": False
            }
    
    def generate_feedback(self, score: float, details: Dict[str, Any]) -> str:
        """スコアに基づいてフィードバックを生成"""
        text_similarity = details.get("text_analysis", {}).get("text_similarity", 0)
        
        feedback_messages = []

        if score >= 90:
            feedback_messages.append("素晴らしい発音です！")
            if text_similarity > 0.95:
                feedback_messages.append("認識精度も非常に高く、完璧に近い発音です。")
            else:
                feedback_messages.append("中国語の音韻的特徴がよく捉えられています。")
        elif score >= 80:
            feedback_messages.append("良好な発音です。")
            if text_similarity > 0.85:
                feedback_messages.append("ほとんど正確に認識されています。")
            else:
                feedback_messages.append("全体的に明瞭で聞き取りやすいです。")
            feedback_messages.append("さらに流暢さを向上させる練習を続けると良いでしょう。")
        elif score >= 70:
            feedback_messages.append("基本的な発音はできています。")
            if text_similarity < 0.75:
                feedback_messages.append("一部の音素や単語が不明瞭な可能性があります。")
            feedback_messages.append("声調の区別をより意識し、発音をはっきりさせる練習をしましょう。")
        elif score >= 60:
            feedback_messages.append("発音の改善が必要です。")
            if text_similarity < 0.65:
                feedback_messages.append("認識精度が低めです。")
            feedback_messages.append("個々の音素と声調の基礎を重点的に練習することをお勧めします。")
        else:
            feedback_messages.append("発音の基礎練習が必要です。")
            if text_similarity < 0.5:
                feedback_messages.append("音声認識が困難なレベルです。")
            feedback_messages.append("ピンインの発音ルールを再確認し、基本から丁寧に練習しましょう。")
            
        # 詳細情報に基づく追加フィードバック
        audio_duration = details.get("audio_features", {}).get("duration", 0)
        mean_energy = details.get("audio_features", {}).get("mean_energy", 0)
        pitch_variation = details.get("audio_features", {}).get("pitch_variation", 0)

        if audio_duration < self.config.min_audio_duration * 1.2 and score < 80:
            feedback_messages.append("もう少し長く、自然に発話することを意識してみましょう。")
        elif audio_duration > self.config.max_audio_duration * 0.8 and score < 80:
            feedback_messages.append("発話が少し長すぎる可能性があります。適切な区切りを意識しましょう。")

        if mean_energy < 0.015:
            feedback_messages.append("もう少し大きな声ではっきりと発音すると、より聞き取りやすくなります。")
        elif mean_energy > 0.2:
            feedback_messages.append("声が大きすぎる、または歪んでいる可能性があります。マイクの位置や音量を確認しましょう。")

        if pitch_variation < 10 and score < 80:
            feedback_messages.append("中国語の声調はピッチの変化が重要です。もっとメリハリをつけて発音してみましょう。")
        elif pitch_variation > 120 and score < 80:
            feedback_messages.append("ピッチの変動が大きすぎるかもしれません。不自然に聞こえないよう調整しましょう。")

        # 重複を排除して結合
        return " ".join(list(dict.fromkeys(feedback_messages)))
    
    def create_fallback_result(self, error_message: str, detailed_errors: List[str] = None) -> Dict[str, Any]:
        """エラー時のフォールバック結果を作成"""
        logger.warning(f"フォールバック実行: {error_message}")
        
        result = {
            "success": False,
            "score": 0.0,
            "feedback": "音声分析に失敗しました。詳細エラーを確認してください。",
            "details": {
                "overall_pronunciation_score": 0.0,
                "intonation_score": 0.0,
                "rhythm_score": 0.0,
                "audio_features": {
                    "duration": 0.0,
                    "mean_energy": 0.0,
                    "pitch_variation": 0.0,
                    "pitch_values_count": 0
                },
                "text_analysis": {
                    "text_similarity": 0.0,
                    "asr_confidence": 0.0
                },
                "asr_segments_available": False
            },
            "error_message": error_message,
            "recognized_text": "認識できませんでした",
            "original_text": "",
            "model_type": "paddlespeech_fallback",
            "timestamp": datetime.now().isoformat()
        }
        
        if detailed_errors:
            result["detailed_errors"] = detailed_errors
            
        return result
    
    def run_analysis(self, audio_path: str, text: str) -> Dict[str, Any]:
        """メイン分析処理"""
        work_dir = None
        
        try:
            # 環境チェック
            env_ok, env_errors = self.check_paddlespeech_environment()
            if not env_ok:
                logger.warning("PaddleSpeech環境が利用できないため、フォールバックモードで動作")
                error_msg = f"PaddleSpeech環境が正しく設定されていません。詳細: {'; '.join(env_errors[:3])}"
                return self.create_fallback_result(error_msg, env_errors)
            
            # 音声ファイル検証
            try:
                self.validate_audio_file(audio_path)
            except Exception as e:
                return self.create_fallback_result(f"音声ファイル検証失敗: {str(e)}")
            
            # 作業ディレクトリ作成
            work_dir = Path(tempfile.mkdtemp(prefix="paddlespeech_analysis_"))
            logger.info(f"作業ディレクトリ作成: {work_dir}")
            
            # 音声前処理
            processed_audio = work_dir / "processed_audio.wav"
            preprocess_ok, preprocess_error = self.preprocess_audio(audio_path, str(processed_audio))
            if not preprocess_ok:
                return self.create_fallback_result(f"音声前処理失敗: {preprocess_error}")
            
            # 音声認識実行
            # ASR結果からsegments情報も取得
            recognized_text, asr_confidence, asr_segments = self.perform_asr(str(processed_audio))
            
            # 発音品質分析
            quality_details = self.analyze_pronunciation_quality(
                str(processed_audio), recognized_text, text, asr_confidence, asr_segments # segmentsを渡す
            )
            
            # 総合スコア計算 (overall_pronunciation_scoreを主要なスコアとする)
            overall_score = quality_details["overall_pronunciation_score"]
            
            # フィードバック生成
            feedback = self.generate_feedback(overall_score, quality_details)
            
            # 結果返却
            result = {
                "success": True,
                "score": overall_score,
                "feedback": feedback,
                "details": quality_details,
                "recognized_text": recognized_text,
                "original_text": text,
                "model_type": "paddlespeech_custom_evaluation", # カスタム評価モデルであることを示す
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"分析完了: スコア={overall_score:.2f}, 認識テキスト='{recognized_text}'")
            return result
            
        except Exception as e:
            logger.error(f"分析中に予期しないエラーが発生しました: {str(e)}", exc_info=True)
            return self.create_fallback_result(f"分析中に予期しないエラーが発生しました: {str(e)}")
            
        finally:
            # クリーンアップ
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.info(f"作業ディレクトリ削除: {work_dir}")
                except Exception as e:
                    logger.error(f"クリーンアップエラー: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='PaddleSpeech音声発音品質分析ツール（AlignExecutorなし）')
    parser.add_argument('--audio', required=True, help='入力音声ファイルパス')
    parser.add_argument('--text', required=True, help='評価対象のテキスト')
    parser.add_argument('--asr-model', default='conformer_wenetspeech-zh-16k', 
                       help='ASRモデル名')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                       help='使用デバイス')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='ターゲットサンプルレート（音声前処理・ASR用）')
    
    args = parser.parse_args()
    
    logger.info(f"コマンドライン引数: audio='{args.audio}', text='{args.text}', "
                f"asr_model='{args.asr_model}', device='{args.device}', sample_rate={args.sample_rate}")
    
    config = PaddleSpeechConfig(
        asr_model=args.asr_model,
        device=args.device,
        target_sample_rate=args.sample_rate,
        asr_sample_rate=args.sample_rate # ASRのサンプルレートも合わせておく
    )
    
    service = PaddleSpeechService(config)
    result = service.run_analysis(args.audio, args.text)
    
    # 結果をJSON形式で標準出力
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()