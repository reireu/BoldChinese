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
    from paddlespeech.cli.vector import VectorExecutor
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

class PaddleSpeechAlignmentError(Exception):
    pass

class PaddleSpeechService:
    def __init__(self, config: PaddleSpeechConfig):
        self.config = config
        self.asr_executor = None
        self.text_executor = None
        
    def check_paddlespeech_environment(self) -> Tuple[bool, List[str]]:
        """PaddleSpeech環境をチェック"""
        logger.info("PaddleSpeech環境をチェック中...")
        errors = []
        
        try:
            if not PADDLESPEECH_AVAILABLE:
                errors.append("PaddleSpeechがインストールされていません。pip install paddlespeech でインストールしてください。")
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
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            if duration < self.config.min_audio_duration:
                raise ValueError(f"音声が短すぎます: {duration:.2f}秒 (最小: {self.config.min_audio_duration}秒)")
                
            if duration > self.config.max_audio_duration:
                raise ValueError(f"音声が長すぎます: {duration:.2f}秒 (最大: {self.config.max_audio_duration}秒)")
                
            # 無音チェック
            if np.max(np.abs(y)) < self.config.silence_threshold:
                raise ValueError("音声が無音または音量が小さすぎます")
                
            logger.info(f"音声ファイル検証完了: {audio_path} (長さ: {duration:.2f}秒, サンプルレート: {sr}Hz)")
            return True
            
        except Exception as e:
            if "音声" in str(e):
                raise
            else:
                raise ValueError(f"音声ファイルの読み込みエラー: {e}")
    
    def preprocess_audio(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """音声ファイルを前処理（リサンプリング、正規化など）"""
        try:
            # librosaで音声を読み込み
            y, sr = librosa.load(input_path, sr=self.config.target_sample_rate)
            
            # モノラルに変換（既にモノラルの場合は何もしない）
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
                
            # 音量正規化
            y = librosa.util.normalize(y)
            
            # 無音部分をトリミング
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # soundfileで保存
            sf.write(output_path, y_trimmed, self.config.target_sample_rate)
            
            output_path_obj = Path(output_path)
            if not output_path_obj.exists():
                return False, f"前処理後のファイルが作成されませんでした: {output_path}"
                
            logger.info(f"音声前処理完了: {output_path}")
            return True, ""
            
        except Exception as e:
            error_msg = f"音声前処理エラー: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def perform_asr(self, audio_path: str) -> Tuple[str, float]:
        """音声認識を実行"""
        try:
            if self.asr_executor is None:
                self.asr_executor = ASRExecutor()
                
            # ASR実行
            result = self.asr_executor(
                audio_file=audio_path,
                model=self.config.asr_model,
                lang=self.config.asr_lang,
                sample_rate=self.config.asr_sample_rate,
                device=self.config.device
            )
            
            # 結果の解析
            if isinstance(result, str):
                recognized_text = result
                confidence = 0.8  # デフォルト信頼度
            elif isinstance(result, dict):
                recognized_text = result.get('text', '')
                confidence = result.get('confidence', 0.8)
            else:
                recognized_text = str(result)
                confidence = 0.8
                
            logger.info(f"ASR結果: {recognized_text} (信頼度: {confidence:.3f})")
            return recognized_text, confidence
            
        except Exception as e:
            error_msg = f"音声認識エラー: {e}"
            logger.error(error_msg)
            raise PaddleSpeechAlignmentError(error_msg)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算"""
        try:
            # 簡単な文字レベルの類似度計算
            if not text1 or not text2:
                return 0.0
                
            # 空白を除去して比較
            text1_clean = text1.replace(' ', '').replace('\n', '').replace('\t', '')
            text2_clean = text2.replace(' ', '').replace('\n', '').replace('\t', '')
            
            if not text1_clean or not text2_clean:
                return 0.0
                
            # 編集距離ベースの類似度
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
                return 1.0
                
            similarity = 1 - (distance / max_len)
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"テキスト類似度計算エラー: {e}")
            return 0.0
    
    def analyze_pronunciation_quality(self, audio_path: str, recognized_text: str, 
                                    original_text: str, asr_confidence: float) -> Dict[str, Any]:
        """発音品質を分析"""
        try:
            # 音声特徴量の抽出
            y, sr = librosa.load(audio_path, sr=self.config.target_sample_rate)
            
            # 基本的な音声特徴量
            duration = len(y) / sr
            rms_energy = librosa.feature.rms(y=y)[0]
            mean_energy = np.mean(rms_energy)
            
            # ピッチ特徴量（中国語の声調分析に重要）
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
                    
            # 発音品質スコアの計算
            base_score = 70.0
            
            # ASR信頼度の影響
            base_score += asr_confidence * 20
            
            # テキスト一致度の影響
            text_similarity = self.calculate_text_similarity(recognized_text, original_text)
            base_score += text_similarity * 10
            
            # 音声品質の影響
            if mean_energy > 0.01:  # 適切な音量
                base_score += 5
            if 0.5 < duration < 10:  # 適切な長さ
                base_score += 5
                
            # ピッチの安定性（中国語の声調）
            if pitch_values:
                pitch_std = np.std(pitch_values)
                if 20 < pitch_std < 100:  # 適度な声調変化
                    base_score += 5
                    
            pronunciation_score = min(100, max(0, base_score))
            
            # 各要素のスコア
            intonation_score = pronunciation_score * (0.9 + 0.1 * len(pitch_values) / max(1, duration * 10))
            rhythm_score = pronunciation_score * (0.95 if 0.5 < duration < 10 else 0.8)
            
            return {
                "pronunciation": round(pronunciation_score, 2),
                "intonation": round(min(100, max(0, intonation_score)), 2),
                "rhythm": round(min(100, max(0, rhythm_score)), 2),
                "duration": round(duration, 2),
                "mean_energy": round(mean_energy, 4),
                "pitch_variation": round(np.std(pitch_values) if pitch_values else 0, 2),
                "text_similarity": round(text_similarity, 3),
                "asr_confidence": round(asr_confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"発音品質分析エラー: {e}")
            return {
                "pronunciation": 50.0,
                "intonation": 50.0,
                "rhythm": 50.0,
                "duration": 0.0,
                "mean_energy": 0.0,
                "pitch_variation": 0.0,
                "text_similarity": 0.0,
                "asr_confidence": asr_confidence
            }
    
    def generate_feedback(self, score: float, details: Dict[str, Any]) -> str:
        """スコアに基づいてフィードバックを生成"""
        text_similarity = details.get("text_similarity", 0)
        
        if score >= 90:
            feedback = "素晴らしい発音です！"
            if text_similarity > 0.9:
                feedback += "認識精度も完璧です。"
            feedback += "中国語の声調も正確に表現されています。"
        elif score >= 80:
            feedback = "良好な発音です！"
            if text_similarity > 0.8:
                feedback += "ほぼ正確に認識されています。"
            feedback += "声調の練習でさらに向上できます。"
        elif score >= 70:
            feedback = "基本的な発音は良好です。"
            if text_similarity < 0.7:
                feedback += "一部の音素が不明瞭です。"
            feedback += "声調の区別を意識して練習しましょう。"
        elif score >= 60:
            feedback = "発音の改善が必要です。"
            if text_similarity < 0.6:
                feedback += "音素の精度向上が必要です。"
            feedback += "基本的な音素と声調から練習しましょう。"
        else:
            feedback = "発音の練習が必要です。"
            if text_similarity < 0.5:
                feedback += "音素の認識精度が低いです。"
            feedback += "ピンインの基礎から始めることをお勧めします。"
            
        return feedback
    
    def create_fallback_result(self, error_message: str, detailed_errors: List[str] = None) -> Dict[str, Any]:
        """エラー時のフォールバック結果を作成"""
        logger.warning(f"フォールバック実行: {error_message}")
        
        result = {
            "success": False,
            "score": 0.0,
            "feedback": "音声分析に失敗しました。PaddleSpeechの設定を確認してください。",
            "details": {
                "pronunciation": 0.0,
                "intonation": 0.0,
                "rhythm": 0.0,
                "duration": 0.0,
                "mean_energy": 0.0,
                "pitch_variation": 0.0,
                "text_similarity": 0.0,
                "asr_confidence": 0.0
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
            work_dir = Path(tempfile.mkdtemp(prefix="paddlespeech_"))
            logger.info(f"作業ディレクトリ作成: {work_dir}")
            
            # 音声前処理
            processed_audio = work_dir / "processed_audio.wav"
            preprocess_ok, preprocess_error = self.preprocess_audio(audio_path, str(processed_audio))
            if not preprocess_ok:
                return self.create_fallback_result(f"音声前処理失敗: {preprocess_error}")
            
            # 音声認識実行
            recognized_text, asr_confidence = self.perform_asr(str(processed_audio))
            
            # 発音品質分析
            quality_details = self.analyze_pronunciation_quality(
                str(processed_audio), recognized_text, text, asr_confidence
            )
            
            # 総合スコア計算
            overall_score = quality_details["pronunciation"]
            
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
                "model_type": "paddlespeech",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"分析完了: スコア={overall_score:.2f}, 認識テキスト='{recognized_text}'")
            return result
            
        except Exception as e:
            logger.error(f"予期しないエラー: {str(e)}", exc_info=True)
            return self.create_fallback_result(f"予期しないエラー: {str(e)}")
            
        finally:
            # クリーンアップ
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.info(f"作業ディレクトリ削除: {work_dir}")
                except Exception as e:
                    logger.error(f"クリーンアップエラー: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='PaddleSpeech音声アライメントツール（中国語対応）')
    parser.add_argument('--audio', required=True, help='音声ファイルパス')
    parser.add_argument('--text', required=True, help='テキスト')
    parser.add_argument('--asr-model', default='conformer_wenetspeech-zh-16k', 
                       help='ASRモデル名')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                       help='使用デバイス')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='サンプルレート')
    
    args = parser.parse_args()
    
    logger.info(f"引数: audio={args.audio}, text={args.text}, "
                f"asr_model={args.asr_model}, device={args.device}")
    
    config = PaddleSpeechConfig(
        asr_model=args.asr_model,
        device=args.device,
        target_sample_rate=args.sample_rate
    )
    
    service = PaddleSpeechService(config)
    result = service.run_analysis(args.audio, args.text)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
