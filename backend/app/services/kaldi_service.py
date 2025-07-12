import subprocess
import os
import json
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt # これは現在のコードでは直接使用されていませんが、将来的な可視化のために残します
from scipy.io.wavfile import write as write_wav # 同上
from scipy.signal import find_peaks # 同上
import warnings
import tempfile
import shutil # ファイル/ディレクトリ削除のため追加
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

@dataclass
class KaldiConfig:
    kaldi_root: str = os.getenv("KALDI_ROOT", "/opt/kaldi")  # デフォルトパスを変更
    wsj_s5_root: str = os.getenv("WSJ_S5_ROOT", "/opt/kaldi/egs/wsj/s5")
    model_dir: str = os.getenv("MANDARIN_MODEL_PATH", "/opt/kaldi/egs/mandarin/s5")
    
class ProcessTimeLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # リクエスト処理時間をログ出力
        logger.info(
            f"Path: {request.url.path} "
            f"Method: {request.method} "
            f"Processing Time: {process_time:.3f}s"
        )
        
        # レスポンスヘッダーに処理時間を追加
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 警告を制御
warnings.filterwarnings('ignore', category=UserWarning)

# --- 定数（計算ロジックの最適化） ---
# _pinyin_to_detailed_phonemesで使用するリストを一度だけソートして定義
INITIALS = sorted(['zh', 'ch', 'sh', 'ng', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                   'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w'],
                  key=len, reverse=True)
FINALS = sorted(['iang', 'iong', 'uang', 'ueng', 'ian', 'iao', 'ing', 'ang',
                 'eng', 'ong', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'in', 'un',
                 'iu', 'ie', 'ui', 'ue', 'er', 'a', 'o', 'e', 'i', 'u', 'v'],
                key=len, reverse=True)

# --- 設定とパラメータ ---
@dataclass
class AnalysisConfig:
    """分析設定のデータクラス"""
    kaldi_root: str = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi")
    mandarin_model_path: str = "" # 例: "{KALDI_ROOT}/egs/mandarin_bn_bci"
    max_text_length: int = 600
    min_text_length: int = 1
    audio_sample_rate: int = 16000
    tone_weight: float = 0.4  # 声調評価の重み
    pronunciation_weight: float = 0.6  # 発音評価の重み
    kaldi_nj_jobs: int = int(os.getenv("KALDI_NJ_JOBS", "2")) # Kaldi並列処理数

    def __post_init__(self):
        if not self.mandarin_model_path:
            self.mandarin_model_path = os.getenv("MANDARIN_MODEL_PATH",
                                               f"{self.kaldi_root}/egs/mandarin_bn_bci")
        if not os.path.exists(self.kaldi_root):
            logger.warning(f"KALDI_ROOT '{self.kaldi_root}' が存在しません。Kaldiは機能しません。")
        if not os.path.exists(self.mandarin_model_path):
            logger.warning(f"MANDARIN_MODEL_PATH '{self.mandarin_model_path}' が存在しません。Kaldiは機能しません。")

# --- カスタム例外クラス ---
class PronunciationAnalysisError(Exception):
    """発音分析プロセス全般で発生するエラーの基底クラス"""
    pass

class KaldiAlignmentError(PronunciationAnalysisError):
    """Kaldi強制アラインメントの実行中に発生するエラー"""
    pass

class AudioProcessingError(PronunciationAnalysisError):
    """音声処理中に発生するエラー"""
    pass

class ValidationError(PronunciationAnalysisError):
    """入力データの検証エラー"""
    pass

# --- KaldiService クラス ---
class KaldiService:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._phones_map_loaded = False
        self._phone_id_to_symbol_map: Dict[int, str] = PHONE_MAP.copy() # デフォルトマップをコピー

    def _load_phones_map_from_file(self) -> None:
        """phones.txt から音素IDマップをロードする"""
        if self._phones_map_loaded:
            return

        phones_file = Path(self.config.kaldi_root) / "data" / "lang" / "phones.txt"
        if not phones_file.exists():
            logger.warning(f"phones.txtが見つかりません: {phones_file}。デフォルトの音素マップを使用します。")
            return

        try:
            with open(phones_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        self._phone_id_to_symbol_map[int(parts[1])] = parts[0]
            self._phones_map_loaded = True
            logger.info(f"phones.txt から {len(self._phone_id_to_symbol_map)} 個の音素をロードしました。")
        except Exception as e:
            logger.error(f"phones.txtのロード中にエラーが発生しました: {e}")
            # エラー時も_phones_map_loadedはFalseのままにして、再試行またはデフォルト利用を可能にする

    def _map_phone_id_to_symbol(self, phone_id: int) -> str:
        """音素IDを音素記号に変換（phones.txtから読み込み）"""
        self._load_phones_map_from_file() # 初回呼び出し時にロードを試みる
        return self._phone_id_to_symbol_map.get(phone_id, f"phone_{phone_id}")

    def _extract_acoustic_features(self, audio_path: str) -> Dict[str, Any]:
        """音響特徴量の抽出（ピッチなど） - 必要なものに絞り込み"""
        try:
            # ターゲットサンプリングレートにリサンプリングしながらロード
            y, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)

            # 音声が空でないことを確認
            if len(y) == 0:
                raise AudioProcessingError(f"音声データが空です: {audio_path}")

            # ピッチ抽出
            # thresholdを調整することで、無音部分のピッチ誤検出を減らせる可能性があります
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.01)
            pitch_values = []
            if pitches.shape[1] > 0:
                for t in range(pitches.shape[1]):
                    # ゼロ除算を避けるため、magnitudesの和が0でないことを確認
                    if magnitudes[:, t].sum() > 0:
                        index = np.argmax(magnitudes[:, t])
                        pitch = pitches[index, t]
                        if pitch > 0: # 無音や非ピッチング箇所のゼロピッチを除外
                            pitch_values.append(pitch)
            
            # 以下は現在のロジックで直接使われていないため、高速化のためにコメントアウト
            # MFCCの抽出 - 現在のスコアリングロジックで直接利用されていない
            # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # スペクトラル特徴量 - 現在のスコアリングロジックで直接利用されていない
            # spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            # spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            duration = len(y) / sr
            rms_energy = np.sqrt(np.mean(y**2)) if len(y) > 0 else 0

            return {
                "pitch_values": pitch_values,
                "mean_pitch": np.mean(pitch_values) if pitch_values else 0,
                "pitch_std": np.std(pitch_values) if pitch_values else 0,
                # "mfccs": mfccs.tolist(), # コメントアウト
                # "spectral_centroids": spectral_centroids.tolist(), # コメントアウト
                # "spectral_rolloff": spectral_rolloff.tolist(), # コメントアウト
                "duration": duration,
                "rms_energy": rms_energy
            }
        except Exception as e:
            logger.error(f"音響特徴量抽出エラー ({audio_path}): {e}", exc_info=True)
            raise AudioProcessingError(f"音響特徴量抽出に失敗しました: {e}")

    def _check_kaldi_availability(self) -> bool:
        """Kaldiの利用可能性をチェック"""
        # Kaldiバイナリの存在確認 + モデルパスの確認
        kaldi_bin_path = Path(self.config.kaldi_root) / "src" / "bin"
        model_path_exists = Path(self.config.mandarin_model_path).exists()
        
        if not kaldi_bin_path.exists():
            logger.warning(f"Kaldiバイナリパス '{kaldi_bin_path}' が見つかりません。")
            return False
        if not model_path_exists:
            logger.warning(f"Kaldiモデルパス '{self.config.mandarin_model_path}' が見つかりません。")
            return False
        
        # 簡易的なスクリプトの存在チェック
        align_script = Path(self.config.kaldi_root) / "steps" / "align_si.sh"
        if not align_script.exists():
            logger.warning(f"Kaldiスクリプト '{align_script}' が見つかりません。")
            return False
        
        logger.info("Kaldiの基本パスとモデルパスが確認されました。")
        return True

    def _run_real_kaldi_alignment_full(self, audio_path: str, text: str) -> Dict[str, Any]:
        """実環境用：Kaldi強制アラインメントの実行（完全版）"""
        # NamedTemporaryDirectoryを使用して、一時ディレクトリを安全に作成し、自動削除
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            logger.info(f"一時Kaldiディレクトリ: {temp_dir}")

            try:
                # 1. データ準備：音声リストなど
                # wav.scp と text を書き出す
                (temp_dir / "wav.scp").write_text(f"utt1 {audio_path}\n")
                (temp_dir / "text").write_text(f"utt1 {text}\n")
                (temp_dir / "utt2spk").write_text("utt1 spk1\n")
                (temp_dir / "spk2utt").write_text("spk1 utt1\n")
                
                # Kaldiのデータ準備スクリプトを実行 (必要に応じて)
                # 一般的な align_si.sh は既にデータディレクトリを期待するため、このステップは通常不要か、
                # ユーザーが別途 'data/local/dict', 'data/lang'などを準備する必要があります。
                # 例: local/prepare_data.sh data/local/dict data/local/lm_train_data data/lang
                # 今回は直接アラインメントに渡す構成なので省略。

                # 2. align_si.sh (triphone align) を実行
                # Kaldiの実行環境によっては、'run.pl'や'cmd.sh'の設定が必要です。
                # ここでは直接コマンドを構築します。
                # --nj で並列処理数を設定
                kaldi_align_cmd = [
                   str(Path(self.config.kaldi_root) / "steps" / "align_si.sh"),
                   "--nj", str(self.config.kaldi_nj_jobs), # 並列ジョブ数
                   "--cmd", "run.pl", # または queue.pl など
                   str(temp_dir),
                   str(Path(self.config.kaldi_root) / "data" / "lang"), # 言語モデルパス
                   str(Path(self.config.kaldi_root) / "exp" / "tri4a"), # 訓練済みモデルパス
                   str(temp_dir / "align_output") # アラインメント結果出力先
                ]
                
                logger.info(f"Kaldi align_si.sh 実行コマンド: {' '.join(kaldi_align_cmd)}")
                
                # subprocess.run の timeout を適切な値に設定
                align_process = subprocess.run(kaldi_align_cmd,
                                               capture_output=True, text=True,
                                               timeout=120, # タイムアウトを適切に設定
                                               check=False) # check=Trueだとエラー時に例外を出す
                
                if align_process.returncode != 0:
                    logger.error(f"Kaldi align_si.sh 実行エラー: {align_process.stderr}")
                    logger.error(f"Kaldi align_si.sh 標準出力: {align_process.stdout}")
                    raise KaldiAlignmentError(f"Kaldi align_si.sh 実行失敗: {align_process.stderr}")
                
                # 3. CTM 出力を得る
                # align_si.sh の出力は align_output/ali.<job_id>.gz にあります
                ali_file = temp_dir / "align_output" / "ali.1.gz" # nj=1の場合
                if not ali_file.exists():
                    # 複数のジョブの場合を考慮（ただしnj=1で実行するためali.1.gzになるはず）
                    # より堅牢な実装では、align_output内のali.*.gzをすべて探し、結合する必要があります。
                    raise KaldiAlignmentError(f"Kaldiアラインメント結果ファイルが見つかりません: {ali_file}")

                ctm_cmd = [
                   str(Path(self.config.kaldi_root) / "src" / "bin" / "ali-to-phones"),
                   "--ctm-output",
                   str(Path(self.config.kaldi_root) / "exp" / "tri4a" / "final.mdl"), # モデルパス
                   f"ark:gunzip -c {ali_file} |"
                ]
                
                logger.info(f"Kaldi ali-to-phones 実行コマンド: {' '.join(ctm_cmd)}")
                # shell=Trueを使う場合は、コマンド全体を文字列として渡す必要がある
                ctm_process = subprocess.run(" ".join(ctm_cmd), shell=True,
                                              capture_output=True, text=True,
                                              timeout=60, check=False)
                
                if ctm_process.returncode != 0:
                    logger.error(f"CTM生成エラー: {ctm_process.stderr}")
                    logger.error(f"CTM生成標準出力: {ctm_process.stdout}")
                    raise KaldiAlignmentError(f"CTM生成失敗: {ctm_process.stderr}")
                
                # 4. CTM をパース
                alignments = self._parse_ctm(ctm_process.stdout)
                
                logger.info(f"Kaldiアラインメント成功。{len(alignments)}個の音素が検出されました。")
                return {"words": alignments}

            except subprocess.TimeoutExpired as e:
                raise KaldiAlignmentError(f"Kaldi実行タイムアウト: {e}")
            except Exception as e:
                logger.error(f"Kaldiアラインメント中に予期せぬエラー: {e}", exc_info=True)
                raise KaldiAlignmentError(f"Kaldiアラインメント失敗: {e}")
            finally:
                # TemporaryDirectoryはwithブロックを抜けると自動削除されるが、念のため
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"一時ディレクトリ {temp_dir} を削除しました。")


    def _generate_realistic_dummy_alignment(self, text: str, acoustic_features: Dict) -> Dict[str, Any]:
        """音響特徴量に基づく現実的なダミーアラインメント生成"""
        duration = acoustic_features.get("duration", 1.0)
        mean_pitch = acoustic_features.get("mean_pitch", 200)
        pitch_std = acoustic_features.get("pitch_std", 20)
        
        words = list(text) # 中国語の文字を単語として扱う
        if not words:
            return {"words": [], "overall_score": 0.0, "acoustic_features": acoustic_features}

        word_duration = duration / len(words)
        
        alignment_words = []
        current_time = 0.0
        
        for i, word in enumerate(words):
            word_end = current_time + word_duration
            
            pinyin = self._estimate_pinyin(word)
            
            phones = self._generate_phones_for_word(word, pinyin, current_time, word_end, 
                                                    mean_pitch, pitch_std)
            
            base_score = self._calculate_acoustic_score(acoustic_features, i, len(words))
            
            for phone in phones:
                phone["score"] = max(0.3, min(1.0, base_score + np.random.normal(0, 0.1)))
            
            alignment_words.append({
                "word": word,
                "start": current_time,
                "end": word_end,
                "pinyin": pinyin,
                "phones": phones
            })
            
            current_time = word_end
        
        overall_score = np.mean([np.mean([p["score"] for p in w["phones"]]) 
                               if w["phones"] else 0 for w in alignment_words]) if alignment_words else 0.0
        
        return {
            "words": alignment_words,
            "overall_score": overall_score,
            "acoustic_features": acoustic_features
        }

    def _estimate_pinyin(self, word: str) -> str:
        """単語からピンインを推定（簡易版） - 実際のアプリでは辞書やライブラリ使用"""
        pinyin_dict = {
            "你": "ni3", "好": "hao3", "我": "wo3", "是": "shi4",
            "的": "de", "在": "zai4", "有": "you3", "不": "bu4",
            "一": "yi1", "人": "ren2", "中": "zhong1", "国": "guo2"
        }
        return pinyin_dict.get(word, "unknown")

    def _generate_phones_for_word(self, word: str, pinyin: str, start_time: float, 
                                end_time: float, mean_pitch: float, pitch_std: float) -> List[Dict]:
        """単語の音素を生成"""
        duration = end_time - start_time
        phones = []
        
        phonemes = self._pinyin_to_detailed_phonemes(pinyin)
        
        if not phonemes:
            return []
        
        phone_duration = duration / len(phonemes)
        current_time = start_time
        
        for i, phoneme in enumerate(phonemes):
            phone_end = current_time + phone_duration
            
            phone_type = self._determine_phone_type(phoneme, i, len(phonemes))
            
            difficulty = self._get_phoneme_difficulty(phoneme)
            base_score = 0.9 - difficulty * 0.3 + np.random.normal(0, 0.1)
            
            phones.append({
                "phone": f"{phoneme}_{phone_type}",
                "start": current_time,
                "end": phone_end,
                "score": max(0.2, min(1.0, base_score)),
                "pitch": mean_pitch + np.random.normal(0, pitch_std * 0.5) if pitch_std else mean_pitch
            })
            
            current_time = phone_end
        
        return phones

    def _pinyin_to_detailed_phonemes(self, pinyin: str) -> List[str]:
        """ピンインを詳細な音素に分解（定数リストを利用）"""
        if pinyin == "unknown":
            return ["unknown"]
        
        clean_pinyin = re.sub(r'[1-5]', '', pinyin)
        tone = re.search(r'[1-5]', pinyin)
        tone_num = tone.group() if tone else "0"
        
        phonemes = []
        remaining = clean_pinyin

        for initial in INITIALS: # 定数リストを利用
            if remaining.startswith(initial):
                phonemes.append(initial)
                remaining = remaining[len(initial):]
                break
        
        if remaining:
            for final in FINALS: # 定数リストを利用
                if remaining.startswith(final):
                    phonemes.append(final)
                    remaining = remaining[len(final):]
                    break
        
        if tone_num != "0":
            phonemes.append(tone_num)
        
        return phonemes if phonemes else ["unknown"]

    def _determine_phone_type(self, phoneme: str, position: int, total: int) -> str:
        """音素のタイプを決定（B=開始, I=中間, E=終了, S=単一）"""
        if total == 1:
            return "S"
        elif position == 0:
            return "B"
        elif position == total - 1:
            return "E"
        else:
            return "I"

    def _get_phoneme_difficulty(self, phoneme: str) -> float:
        """音素の難易度を返す（0.0-1.0）"""
        difficult_phonemes = {
            'zh': 0.8, 'ch': 0.8, 'sh': 0.7, 'r': 0.9, 'j': 0.6, 'q': 0.6, 'x': 0.6,
            'z': 0.7, 'c': 0.7, 's': 0.5, 'ng': 0.8, 'er': 0.9
        }
        return difficult_phonemes.get(phoneme, 0.3)

    def _calculate_acoustic_score(self, acoustic_features: Dict, word_index: int, total_words: int) -> float:
        """音響特徴量に基づくスコア計算"""
        base_score = 0.8
        
        rms_energy = acoustic_features.get("rms_energy", 0.1)
        if rms_energy < 0.01:
            base_score -= 0.3
        elif rms_energy > 0.5:
            base_score -= 0.1
        
        pitch_std = acoustic_features.get("pitch_std", 0)
        if pitch_std > 50:
            base_score -= 0.2
        
        return max(0.3, min(1.0, base_score))

    def _calculate_score_from_pitch(self, pitch: float) -> float:
        """ピッチから音素スコアを計算"""
        if pitch == 0:
            return 0.5
        
        if 80 <= pitch <= 400:
            return 0.9
        elif 50 <= pitch < 80 or 400 < pitch <= 500:
            return 0.7
        else:
            return 0.4

    def _parse_ctm(self, ctm_text: str) -> List[Dict[str, Any]]:
        """CTM フォーマットを解析し、phone 単位の align 情報を抽出"""
        align_list = []
        for line in ctm_text.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                # utt, ch, start, dur, phone_id, confidence, word_id...
                utt, ch, start, dur, phone_id_str = parts[:5]
                start, dur = float(start), float(dur)
                phone_id = int(phone_id_str)
                phone_symbol = self._map_phone_id_to_symbol(phone_id)
                
                # CTMにconfidenceが含まれている場合があるが、Kaldiのali-to-phonesでは通常含まれない
                # ここでは暫定的に1.0としておくか、別途推定する
                confidence = float(parts[5]) if len(parts) > 5 and re.match(r'^\d+\.?\d*$', parts[5]) else 1.0

                align_list.append({
                    "phone": phone_symbol,
                    "start": start,
                    "end": start + dur,
                    "score": confidence, # Kaldiから取得したconfidenceを優先
                })
        return align_list

    def _merge_alignment_and_features(self, phones: List[Dict], acoustic_features: Dict) -> Dict:
        """アラインメントと音響特徴量を統合しスコア算出"""
        pitch_values = acoustic_features.get("pitch_values", [])
        duration = acoustic_features.get("duration", 1.0)
        
        if not pitch_values:
            logger.warning("ピッチデータが不足しているため、ピッチに基づくスコアリングができません。")
            # ピッチデータがない場合のフォールバックとして、元のスコアをそのまま利用
            for p in phones:
                p["pitch"] = 0
                if "score" not in p:
                    p["score"] = 0.5 # デフォルトスコアを設定
            return {"phones": phones}

        for p in phones:
            # 該当時間における平均ピッチを計算
            # 音声の総時間に対するセグメントの割合でピッチ配列のインデックスを計算
            start_sec = p["start"]
            end_sec = p["end"]

            # ピッチ配列のフレームレートを考慮してインデックスを計算
            # librosa.piptrackのデフォルトhop_lengthは512、sr=16000の場合、フレームレートは16000/512 = 31.25 frames/sec
            # 正確なフレームレートはlibrosaの内部設定に依存するため、ここでは簡略化
            frame_rate = len(pitch_values) / duration if duration > 0 else 0

            if frame_rate > 0:
                start_idx = int(start_sec * frame_rate)
                end_idx = int(end_sec * frame_rate)
                
                segment_pitches = [
                    pv for i, pv in enumerate(pitch_values)
                    if start_idx <= i < end_idx and pv > 0 # ピッチが0でない有効な値のみ
                ]
            else:
                segment_pitches = []

            segment_mean_pitch = np.mean(segment_pitches) if segment_pitches else 0
            p["pitch"] = segment_mean_pitch
            
            # 既存のスコア（Kaldiから取得したもの）とピッチスコアを統合
            original_kaldi_score = p.get("score", 0.5) # Kaldiからのスコアがない場合のデフォルト
            pitch_score = self._calculate_score_from_pitch(p["pitch"])
            
            # ここで重み付けを調整して最終スコアを決定
            # Kaldiの信頼性とピッチの正確性をバランスさせる
            p["score"] = (original_kaldi_score * 0.7 + pitch_score * 0.3)
            p["score"] = max(0.2, min(1.0, p["score"])) # スコアを0.2-1.0の範囲に正規化

        return {"phones": phones}

    def run_analysis(self, audio_content: bytes, text: str, pinyin_list: List[str]) -> Dict[str, Any]:
        """
        音声とテキスト、ピンインリストに基づいて発音分析を実行するメイン関数。
        処理時間を短縮するため、一時ファイルの効率的な管理とKaldiの並列処理を利用。
        """
        validate_inputs(audio_content, text, pinyin_list)

        # 一時ディレクトリの作成と音声ファイルの書き込み
        # TemporaryDirectoryはwithブロックを抜けると自動削除される
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            audio_path = temp_dir / "input_audio.wav"
            
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_content)
                logger.info(f"音声ファイルを一時パスに保存しました: {audio_path}")

                # 音声ファイル検証 (librosa.loadが内部で呼ばれるため、エラー時はAudioProcessingErrorをRaise)
                is_valid, msg = validate_audio_file(str(audio_path))
                if not is_valid:
                    raise ValidationError(f"音声ファイルの検証エラー: {msg}")

                # 音響特徴量の抽出
                acoustic_features = self._extract_acoustic_features(str(audio_path))
                if not acoustic_features:
                    raise AudioProcessingError("音響特徴量の抽出に失敗しました。")
                logger.info("音響特徴量の抽出が完了しました。")

                # Kaldi強制アラインメントの実行
                # Kaldiが利用できない場合はダミーデータを生成
                if self._check_kaldi_availability():
                    logger.info("Kaldiを利用してアラインメントを実行します。")
                    alignment_result = self._run_real_kaldi_alignment_full(str(audio_path), text)
                else:
                    logger.warning("Kaldiが利用できないため、ダミーアラインメントを生成します。")
                    alignment_result = self._generate_realistic_dummy_alignment(text, acoustic_features)

                # アラインメント結果と音響特徴量の統合
                # Kaldiからの出力は音素レベルのアラインメントが想定される
                if "words" in alignment_result and alignment_result["words"]:
                    # Kaldi出力は通常flatなphoneリストなので、ここでは単語構造を構築し直す
                    # もしKaldiが単語レベルのアラインメントも出すなら、それを利用する
                    # 現状の_run_real_kaldi_alignment_fullはCTM (phone level)を返しているので
                    # 単語への再構築はここで行う必要があります。これは複雑なので、一旦簡略化
                    # ここではalignment_result["words"]が、直接phoneリストであると仮定
                    all_phones = alignment_result["words"]
                    
                    # 音響特徴量を統合してスコアを更新
                    integrated_phones_data = self._merge_alignment_and_features(all_phones, acoustic_features)
                    final_phones = integrated_phones_data["phones"]

                    # 単語レベルのスコアと構造の再構築（簡略版）
                    # 実際のテキストとアラインメントされた音素をマッピングして単語を構築
                    # この部分が最も複雑で、本格的なシステムでは別途詳細なロジックが必要
                    # 今回はCTMが音素リストとして返ってくることを前提に、
                    # pinyin_listと文字数から簡易的にwordを再構成します。
                    word_alignments = []
                    phone_idx = 0
                    for char_idx, char_text in enumerate(list(text)):
                        # ここでpinyin_list[char_idx]を使って、そのピンインが持つ音素数を推定し、
                        # final_phonesから対応する音素を割り当てる必要があります。
                        # ダミー生成の_pinyin_to_detailed_phonemes関数を再利用して音素数を取得します。
                        estimated_phonemes = self._pinyin_to_detailed_phonemes(pinyin_list[char_idx])
                        
                        # 音素が割り当てられない場合のフォールバック
                        num_expected_phones = len(estimated_phonemes) if estimated_phonemes else 1

                        word_phones = []
                        start_time_word = final_phones[phone_idx]['start'] if phone_idx < len(final_phones) else 0.0
                        
                        for _ in range(num_expected_phones):
                            if phone_idx < len(final_phones):
                                word_phones.append(final_phones[phone_idx])
                                phone_idx += 1
                            else:
                                break # 音素が足りない場合

                        end_time_word = word_phones[-1]['end'] if word_phones else start_time_word
                        word_score = np.mean([p['score'] for p in word_phones]) if word_phones else 0.0

                        word_alignments.append({
                            "word": char_text,
                            "pinyin": pinyin_list[char_idx],
                            "start": start_time_word,
                            "end": end_time_word,
                            "phones": word_phones,
                            "score": word_score # 単語スコアも追加
                        })
                    
                    # 声調評価
                    tone_evaluation_result = _evaluate_tone_advanced(pinyin_list, word_alignments)
                    logger.info("声調評価が完了しました。")

                    # 発音評価 (音素スコアに基づく)
                    pronunciation_scores = [p["score"] for w in word_alignments for p in w["phones"]]
                    overall_pronunciation_score = np.mean(pronunciation_scores) if pronunciation_scores else 0.0
                    
                    # 総合評価
                    final_overall_score = (tone_evaluation_result["overall_tone_score"] * self.config.tone_weight +
                                        overall_pronunciation_score * self.config.pronunciation_weight)
                    
                    logger.info("総合評価が完了しました。")

                    return {
                        "overall_score": round(final_overall_score, 3),
                        "overall_pronunciation_score": round(overall_pronunciation_score, 3),
                        "tone_evaluation": tone_evaluation_result,
                        "word_alignments": word_alignments,
                        "acoustic_features_summary": {
                            "duration": acoustic_features.get("duration"),
                            "mean_pitch": acoustic_features.get("mean_pitch"),
                            "pitch_std": acoustic_features.get("pitch_std"),
                            "rms_energy": acoustic_features.get("rms_energy")
                        }
                    }
                else:
                    raise KaldiAlignmentError("アラインメント結果から有効な音素データが見つかりませんでした。")

            except (ValidationError, AudioProcessingError, KaldiAlignmentError) as e:
                logger.error(f"分析処理中にエラーが発生しました: {e}")
                raise
            except Exception as e:
                logger.critical(f"予期せぬクリティカルエラーが発生しました: {e}", exc_info=True)
                raise PronunciationAnalysisError(f"分析処理中に予期せぬエラーが発生しました: {e}")

# --- 入力検証関数 (クラス外で定義されているためそのまま) ---
def validate_inputs(audio_content: bytes, text: str, pinyin_list: List[str]) -> None:
    """入力データの検証を行う"""
    if not audio_content:
        raise ValidationError("音声データが空です")
    
    if not text or not text.strip():
        raise ValidationError("テキストが空です")
    
    text_length = len(text.strip())
    # 設定ファイルから取得
    if not (config.min_text_length <= text_length <= config.max_text_length):
        raise ValidationError(
            f"テキスト長が範囲外です。現在: {text_length}, "
            f"許可範囲: {config.min_text_length}-{config.max_text_length}"
        )
    
    if not pinyin_list:
        raise ValidationError("ピンインリストが空です")

def validate_audio_file(audio_path: str) -> Tuple[bool, str]:
    """音声ファイルの検証"""
    try:
        if not Path(audio_path).exists():
            return False, "音声ファイルが存在しません"
        
        y, sr = librosa.load(audio_path, sr=None) # 元のサンプリングレートでロード
        if len(y) == 0:
            return False, "音声データが空です"
        
        duration = len(y) / sr
        if duration < 0.1:
            return False, "音声が短すぎます（0.1秒未満）"
        
        if duration > 300.0:
            return False, "音声が長すぎます（300秒超）"
        
        return True, "OK"
    except Exception as e:
        return False, f"音声ファイル検証エラー: {str(e)}"

# --- 評価関数 (クラス外で定義されているためそのまま) ---
# この部分は _evaluate_tone_advanced などの関数がKaldiServiceクラスの外で定義されているため、
# そのまま残しますが、もしこれらの関数がKaldiServiceの内部状態にアクセスする必要があるなら、
# クラスのメソッドとして移動することを検討してください。
# 現在のところ、_evaluate_tone_advancedはaligned_words_dataという引数でデータを渡されているため、
# そのままでも動作します。

def _evaluate_tone_advanced(pinyin_with_tones: List[str], 
                          aligned_words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """改良された声調評価"""
    tone_scores = []
    tone_details = []
    
    for i, pinyin in enumerate(pinyin_with_tones):
        word_tone_evaluation = _evaluate_single_tone(pinyin, aligned_words_data, i)
        tone_scores.append(word_tone_evaluation["score"])
        tone_details.append(word_tone_evaluation)
    
    overall_tone_score = np.mean(tone_scores) if tone_scores else 0.0
    
    suggestions = _generate_tone_suggestions(tone_details)
    
    return {
        "word_tone_scores": tone_details,
        "overall_tone_score": round(overall_tone_score, 3),
        "tone_distribution": _analyze_tone_distribution(tone_details),
        "improvement_suggestions": suggestions
    }

def _evaluate_single_tone(pinyin: str, aligned_words: List[Dict], word_index: int) -> Dict[str, Any]:
    """単一音素の声調評価"""
    expected_tone = re.search(r'[1-5]', pinyin)
    expected_tone_num = int(expected_tone.group()) if expected_tone else 0
    
    base_evaluation = {
        "pinyin": pinyin,
        "expected_tone": expected_tone_num,
        "score": 0.5,
        "confidence": 0.0,
        "pitch_pattern": "unknown"
    }
    
    if word_index >= len(aligned_words):
        return base_evaluation
    
    word_data = aligned_words[word_index]
    # ここで音素のリストが word_data["phones"] にあることを前提とします
    # CTM出力からは直接ピンインの音素（声調音素を含む）が来るわけではないため、
    # Kaldiが声調を音素として認識するモデルを使っているか、または追加の解析が必要です。
    # ダミーアラインメントでは 'X_S' のように_typeがつくため、調整が必要です。
    # ここでは、phoneが声調番号で始まるものを声調音素と仮定します。
    tone_phones = [p for p in word_data.get("phones", []) 
                  if re.match(r'[1-5]_', p.get("phone", ""))] # 例: '1_S', '3_B' など
    
    if not tone_phones:
        base_evaluation["score"] = 0.3
        return base_evaluation
    
    recognized_tones = [int(p["phone"][0]) for p in tone_phones if p["phone"][0].isdigit()]
    if not recognized_tones:
        base_evaluation["score"] = 0.3
        return base_evaluation
    
    tone_accuracy = sum(1 for t in recognized_tones if t == expected_tone_num) / len(recognized_tones)
    
    pitch_values = [p.get("pitch", 0) for p in tone_phones if p.get("pitch")]
    pitch_pattern = _analyze_pitch_pattern(pitch_values, expected_tone_num)
    
    tone_confidence = np.mean([p.get("score", 0) for p in tone_phones])
    combined_score = (tone_accuracy * 0.6 + tone_confidence * 0.4)
    
    return {
        "pinyin": pinyin,
        "expected_tone": expected_tone_num,
        "recognized_tones": recognized_tones,
        "score": round(combined_score, 3),
        "confidence": round(tone_confidence, 3),
        "pitch_pattern": pitch_pattern,
        "accuracy": round(tone_accuracy, 3)
    }

def _analyze_pitch_pattern(pitch_values: List[float], expected_tone: int) -> str:
    """ピッチパターンの分析"""
    if len(pitch_values) < 2:
        return "insufficient_data"
    
    # 実際はより洗練されたピッチパターン分析アルゴリズムが必要ですが、ここでは簡略化
    pitch_values_filtered = [p for p in pitch_values if p > 0] # 0Hzを除外
    if len(pitch_values_filtered) < 2:
        return "insufficient_valid_data"

    pitch_diffs = np.diff(pitch_values_filtered)
    
    # ピッチの変化の傾向
    mean_diff = np.mean(pitch_diffs)
    std_dev_pitch = np.std(pitch_values_filtered)

    if expected_tone == 1:  # 高平調
        if std_dev_pitch < 15 and abs(mean_diff) < 5:
            return "correct_flat"
        else:
            return "incorrect_unstable"
    elif expected_tone == 2:  # 上昇調
        if mean_diff > 5 and std_dev_pitch > 10:
            return "correct_rising"
        else:
            return "incorrect_pattern"
    elif expected_tone == 3:  # 上昇下降調 (谷型)
        # より複雑な検出ロジックが必要。簡略化のためにピッチの変動幅を見る
        # ピッチが一度下がり、その後上がるパターンを検出
        min_idx = np.argmin(pitch_values_filtered)
        if min_idx > 0 and min_idx < len(pitch_values_filtered) - 1:
            if pitch_values_filtered[min_idx] < pitch_values_filtered[0] and \
               pitch_values_filtered[min_idx] < pitch_values_filtered[-1]:
                return "correct_dip"
        return "incorrect_pattern"
    elif expected_tone == 4:  # 下降調
        if mean_diff < -5 and std_dev_pitch > 10:
            return "correct_falling"
        else:
            return "incorrect_pattern"
    
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
        # スコアが0でないものから最小値を選択
        non_zero_scores = {k: v for k, v in tone_avg_scores.items() if v > 0}
        if non_zero_scores:
            most_difficult_tone = min(non_zero_scores.items(), key=lambda x: x[1])[0]
        else:
            # 全て0の場合、頻度が最も高い声調を返すなど、フォールバックロジックが必要
            most_difficult_tone = max(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None

    best_tone = None
    if tone_avg_scores:
        non_zero_scores = {k: v for k, v in tone_avg_scores.items() if v > 0}
        if non_zero_scores:
            best_tone = max(non_zero_scores.items(), key=lambda x: x[1])[0]
        else:
             best_tone = min(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else None # 全て0の場合、最小のものを返すなど

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
        
        if score < 0.6: # スコアが低い場合に提案
            pinyin_str = detail['pinyin']
            
            if expected_tone == 1:
                if "unstable" in pitch_pattern:
                    suggestions.append(f"'{pinyin_str}' の第1声: ピッチをより平坦に保つように意識しましょう。音程の変動を抑えてください。")
                elif "incorrect" in pitch_pattern:
                    suggestions.append(f"'{pinyin_str}' の第1声: 高く、平らに発音する練習をしましょう。")
            elif expected_tone == 2:
                if "falling" in pitch_pattern or "incorrect" in pitch_pattern:
                    suggestions.append(f"'{pinyin_str}' の第2声: 音程をはっきりと上昇させるように練習しましょう。質問のイントネーションをイメージしてください。")
            elif expected_tone == 3:
                if "incorrect" in pitch_pattern:
                    suggestions.append(f"'{pinyin_str}' の第3声: 音程を一度下げてから再び上げる「谷型」のパターンを練習しましょう。自然なディップを意識してください。")
            elif expected_tone == 4:
                if "rising" in pitch_pattern or "incorrect" in pitch_pattern:
                    suggestions.append(f"'{pinyin_str}' の第4声: 音程を上から下へ一気に下降させるように発音しましょう。強く断定するイメージです。")
            else:
                # 0声などの不明な声調、またはパターンが特定できない場合
                suggestions.append(f"'{pinyin_str}' の声調: 発音の明瞭度を上げるために、ピッチの安定性や音素の明確さを意識しましょう。")

    # 重複する提案を削除してユニークにする
    return list(sorted(set(suggestions)))

    # 設定をロード
    analysis_config = AnalysisConfig()
    kaldi_service = KaldiService(analysis_config)

    # ダミー音声データとテキスト
    # 実際の音声ファイルパスを使用することも可能
    dummy_audio_path = "test_audio.wav"
    sample_rate = analysis_config.audio_sample_rate
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 簡易的な音声波形 (サイン波)
    y_dummy = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    y_dummy_int16 = (y_dummy * 32767).astype(np.int16)
    
    # WAVファイルとして保存（Kaldiがファイルパスを要求するため）
    write_wav(dummy_audio_path, sample_rate, y_dummy_int16)
    
    with open(dummy_audio_path, "rb") as f:
        audio_bytes = f.read()

    test_text = "你好中国"
    test_pinyin = ["ni3", "hao3", "zhong1", "guo2"]

    print(f"--- 音声分析開始 ---")
    start_time = time.time()
    try:
        result = kaldi_service.run_analysis(audio_bytes, test_text, test_pinyin)
        end_time = time.time()
        print(f"--- 音声分析完了 --- 処理時間: {end_time - start_time:.2f}秒")
        print(json.dumps(result, indent=2, ensure_ascii=False))

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
    
    # 別のダミー音声 (短め)
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
        result_short = kaldi_service.run_analysis(audio_bytes_short, test_text_short, test_pinyin_short)
        end_time_short = time.time()
        print(f"--- 音声分析完了 --- 処理時間: {end_time_short - start_time_short:.2f}秒")
        print(json.dumps(result_short, indent=2, ensure_ascii=False))
    except PronunciationAnalysisError as e:
        print(f"分析エラー (短文): {e}")
    except Exception as e:
        print(f"予期せぬエラー (短文): {e}")
 #   finally:
#        if os.path.exists(dummy_audio_path_short):
 #           os.remove(dummy_audio_path_short)
  #          print(f"一時ファイル {dummy_audio_path_short} を削除しました。")