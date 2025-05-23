import subprocess
import os
import json
import re
import Levenshtein # Levenshtein距離計算のため
import numpy as np # DPテーブルのため
from typing import Dict, Any, List, Optional
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav # ダミー音声ファイル作成用

# --- 環境変数の設定 ---
KALDI_ROOT = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi") 
# MANDARIN_MODEL_PATH の定義ミスの可能性を修正
MANDARIN_MODEL_PATH = os.getenv("MANDARIN_MODEL_PATH", f"{KALDI_ROOT}/egs/mandarin_bn_bci")

# ユーザー入力の制限
MAX_TEXT_LENGTH = 600  
MIN_TEXT_LENGTH = 1  

# --- カスタム例外クラス ---
class PronunciationAnalysisError(Exception):
    """発音分析プロセス全般で発生するエラーの基底クラス"""
    pass

class KaldiAlignmentError(PronunciationAnalysisError):
    """Kaldi強制アラインメントの実行中に発生するエラー"""
    pass

# --- Kaldi関連関数 ---
def _run_kaldi_alignment(audio_path: str, text: str) -> Dict[str, Any]:
    """
    Kaldi を使用して音声ファイルとテキストの強制アラインメントを実行し、
    音素レベルの情報（発音時間、スコアなど）を取得する。
    
    Args:
        audio_path (str): 処理する音声ファイルのパス。
        text (str): 音声に対応する参照テキスト（中国語）。

    Returns:
        Dict[str, Any]: アラインメント結果を含む辞書。
                        例: {"words": [{"word": "你好", "start": 0.1, "end": 0.5, "phones": [...]}]}
    Raises:
        KaldiAlignmentError: Kaldiの実行に失敗した場合。
    """
    # 以下は、KaldiのアライナーがJSON形式で結果を返すことを想定したダミー実装です。
    # 実際のKaldiの出力は、フレーム単位のアラインメント情報や発音の確信度など、
    # より詳細なデータを提供します。
    
    # 仮のKaldi出力構造（これを実際のKaldi出力に合わせて解析する必要があります）
    # 可視化のために、音素の開始・終了時刻とスコアをダミーで追加
    # 最終的にはKaldiから得られる音素ごとの確信度スコア (log-likelihoodなど) を利用できるといいらしい。
    dummy_alignment_output = {
        "words": [
            {"word": "你", "start": 0.0, "end": 0.35, "pinyin": "ni3",
             "phones": [
                 {"phone": "n_B", "start": 0.0, "end": 0.1, "score": 0.95},
                 {"phone": "i_I", "start": 0.1, "end": 0.3, "score": 0.92},
                 {"phone": "3_T", "start": 0.3, "end": 0.35, "score": 0.88}
             ]},
            {"word": "好", "start": 0.35, "end": 0.7, "pinyin": "hao3",
             "phones": [
                 {"phone": "h_B", "start": 0.35, "end": 0.45, "score": 0.90},
                 {"phone": "a_I", "start": 0.45, "end": 0.60, "score": 0.85},
                 {"phone": "o_I", "start": 0.60, "end": 0.65, "score": 0.80},
                 {"phone": "3_T", "start": 0.65, "end": 0.7, "score": 0.75}
             ]}
        ],
        "overall_score": 0.95 # 全体の音声認識の確信度など
    }

    print(f"Running dummy Kaldi alignment for: {audio_path} with text: {text}")
    # 実際のKaldi呼び出しの代わりにダミーデータを返す
    # try:
    #     # 例: Kaldiアラインメントスクリプトの実行
    #     # このスクリプトは、入力音声とテキストからアラインメント結果をJSON形式で出力すると仮定
    #     command = [
    #         "python", "your_kaldi_aligner_wrapper.py", # カスタムラッパーまたはMFA/gentleのCLI
    #         "--audio", audio_path,
    #         "--text", text,
    #         "--model", MANDARIN_MODEL_PATH
    #     ]
    #     result = subprocess.run(
    #         command,
    #         capture_output=True,
    #         text=True,
    #         check=True
    #     )
    #     alignment_data = json.loads(result.stdout)
    #     return alignment_data
    # except subprocess.CalledProcessError as e:
    #     print(f"Kaldi alignment failed: {e.stderr}")
    #     raise KaldiAlignmentError(f"Kaldi alignment process failed: {e.stderr.strip()}")
    # except json.JSONDecodeError:
    #     print(f"Failed to parse Kaldi alignment output as JSON: {result.stdout}")
    #     raise KaldiAlignmentError("Kaldi alignment output was not valid JSON.")
    
    return dummy_alignment_output

# --- 評価関数 ---
def _evaluate_tone(pinyin_with_tones: List[str], aligned_words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    声調の評価を行う。
    アラインメント結果から声調の確信度や逸脱を推定する。
    Kaldiの出力が声調に関する情報（例: tone GMMスコア）を含む場合に有効。

    Args:
        pinyin_with_tones (List[str]): 参照テキストのピンイン（声調付き）。例: ["ni3", "hao3"]
        aligned_words_data (List[Dict[str, Any]]): Kaldiから得られた単語レベルのアラインメント情報。
                                             各単語に音素のリストとスコアが含まれることを想定。

    Returns:
        Dict[str, Any]: 各単語の声調評価スコア、全体的な声調スコアなど。
    """
    tone_scores = []
    
    for i, pinyin in enumerate(pinyin_with_tones):
        word_tone_score = 0.0
        if i < len(aligned_words_data) and "phones" in aligned_words_data[i]:
            # _evaluate_tone() の正規表現に注意の改善: re.fullmatch を使用
            has_tone_phone = any(re.fullmatch(r'[1-5]_T', phone_data.get("phone", "")) for phone_data in aligned_words_data[i]["phones"])
            
            if has_tone_phone:
                relevant_tone_phones_scores = [
                    phone_data["score"] for phone_data in aligned_words_data[i]["phones"]
                    if re.fullmatch(r'[1-5]_T', phone_data.get("phone", ""))
                ]
                if relevant_tone_phones_scores:
                    avg_tone_score_from_kaldi = sum(relevant_tone_phones_scores) / len(relevant_tone_phones_scores)
                    # Kaldiのスコアを線形的にマッピングする
                    word_tone_score = max(0.0, min(1.0, (avg_tone_score_from_kaldi - 0.7) * 2.0 + 0.5)) 
                else:
                    word_tone_score = 0.5 # 声調音素が見つからない場合は中程度のスコア（モデルが提供しない場合など）
            else:
                word_tone_score = 0.2 # 声調音素がアラインされていない場合は低スコア
        
        tone_scores.append({"pinyin": pinyin, "score": round(word_tone_score, 3)})
            
    overall_tone_score = sum([s["score"] for s in tone_scores]) / len(tone_scores) if tone_scores else 0.0
    
    return {
        "word_tone_scores": tone_scores,
        "overall_tone_score": round(overall_tone_score, 3)
    }

def _levenshtein_alignment_with_dp(ref: List[str], hyp: List[str]):
    """
    Levenshtein距離とDPテーブルを計算する。
    Args:
        ref (List[str]): 参照音素列 (reference)
        hyp (List[str]): 仮説音素列 (hypothesis, アラインされた音素)
    Returns:
        Tuple[int, np.ndarray]: Levenshtein距離とDPテーブル
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i  # 参照のみ残る = 削除コスト
    for j in range(m + 1):
        dp[0][j] = j  # 仮説のみ残る = 挿入コスト

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1 # 置換コスト
            dp[i][j] = min(dp[i-1][j] + 1,    # 削除 (Deletion)
                           dp[i][j-1] + 1,    # 挿入 (Insertion)
                           dp[i-1][j-1] + cost)  # 置換 (Substitution) or 一致 (Match)

    return dp[n][m], dp  # 距離とDPテーブル

def _traceback_levenshtein_path(dp_table: np.ndarray, ref: List[str], hyp: List[str]) -> List[Dict[str, Any]]:
    """
    Levenshtein DPテーブルをトレースバックしてアラインメントパスを生成する。
    Args:
        dp_table (np.ndarray): DPテーブル
        ref (List[str]): 参照音素列
        hyp (List[str]): 仮説音素列
    Returns:
        List[Dict[str, Any]]: アラインメントパスのリスト。
                               例: [{"type": "match", "ref_phone": "n", "hyp_phone": "n", "score": 0.9}]
    """
    i, j = dp_table.shape[0] - 1, dp_table.shape[1] - 1
    path = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            # 一致
            path.append({"type": "match", "ref_phone": ref[i-1], "hyp_phone": hyp[j-1]})
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp_table[i][j] == dp_table[i-1][j-1] + 1:
            # 置換
            path.append({"type": "substitution", "ref_phone": ref[i-1], "hyp_phone": hyp[j-1]})
            i -= 1
            j -= 1
        elif i > 0 and dp_table[i][j] == dp_table[i-1][j] + 1:
            # 削除
            path.append({"type": "deletion", "ref_phone": ref[i-1], "hyp_phone": None})
            i -= 1
        elif j > 0 and dp_table[i][j] == dp_table[i][j-1] + 1:
            # 挿入  
            path.append({"type": "insertion", "ref_phone": None, "hyp_phone": hyp[j-1]})
            j -= 1
        else:
            #　不正な状態のエラーハンドル
            raise PronunciationAnalysisError("Levenshtein traceback error: No valid path found.")
    
    return path[::-1] # パスを逆順にする

def _evaluate_pronunciation(aligned_phones_details: List[Dict[str, Any]], reference_phonetic_segments: List[str]) -> Dict[str, Any]:
    """
    発音（音素レベル）の評価を行う。
    Levenshtein距離を使用して、参照とアラインされた音素列の間の距離を測定し、スコア化する。
    
    Args:
        aligned_phones_details (List[Dict[str, Any]]): Kaldiから得られた音素レベルのアラインメント情報。
                                             各音素にスコアや特性が含まれることを想定。
                                             例: [{"phone": "n_B", "start": 0.0, "end": 0.1, "score": 0.95}]
        reference_phonetic_segments (List[str]): 参照テキストを音素に分解したリスト（声調なし）。
                                              例: ["n", "i", "h", "a", "o"]

    Returns:
        Dict[str, Any]: 各音素の発音スコア、全体的な発音スコアなど。
    """
    pronunciation_scores = []
    
    # アラインされた音素の「純粋な」音素名リストと、そのスコアを保持する
    aligned_pure_phones = []
    aligned_phone_raw_scores = [] # Kaldiからの生の音素スコア
    for phone_data in aligned_phones_details:
        pure_phone = phone_data.get("phone", "").split('_')[0]
        # 声調音素やサイレンス音素は除外して発音評価の対象とする
        if pure_phone and not re.fullmatch(r'[1-5]', pure_phone) and pure_phone != 'sil':
            aligned_pure_phones.append(pure_phone)
            aligned_phone_raw_scores.append(phone_data.get("score", 0.0))

    # Levenshtein距離とDPテーブルを計算
    distance, dp_table = _levenshtein_alignment_with_dp(reference_phonetic_segments, aligned_pure_phones)
    
    # 全体スコアの算出
    max_len = max(len(reference_phonetic_segments), len(aligned_pure_phones))
    overall_pronunciation_score = 1.0 - (distance / max_len if max_len > 0 else 0.0)

    # アラインメントパスのトレースバック
    alignment_path = _traceback_levenshtein_path(dp_table, reference_phonetic_segments, aligned_pure_phones)

    # 各参照音素のスコアと評価詳細を決定
    ref_idx_map = {phone: idx for idx, phone in enumerate(reference_phonetic_segments)}
    hyp_idx_map = {phone: idx for idx, phone in enumerate(aligned_pure_phones)}

    # 各参照音素に対応する評価を初期化
    phone_evaluation_details = {phone: {"score": overall_pronunciation_score, "type": "unmatched"} for phone in reference_phonetic_segments}

    # アラインメントパスに基づいて評価を更新
    for path_entry in alignment_path:
        ref_phone = path_entry.get("ref_phone")
        hyp_phone = path_entry.get("hyp_phone")
        entry_type = path_entry["type"]

        if ref_phone: # 参照音素がある場合のみ評価対象
            base_score = overall_pronunciation_score # 全体スコアを初期ベースとする

            if entry_type == "match":
                # マッチした場合、Kaldiの個別スコアと全体スコアを重み付け
                hyp_phone_idx = aligned_pure_phones.index(hyp_phone) # 仮説音素のインデックスを取得
                kaldi_raw_score = aligned_phone_raw_scores[hyp_phone_idx] if hyp_phone_idx < len(aligned_phone_raw_scores) else 0.0
                
                # 音素の重み付きスコアリングを適用
                # ここではKaldiの個別スコアを7割、全体スコアを3割でブレンド
                phone_score = (0.7 * kaldi_raw_score) + (0.3 * overall_pronunciation_score)
                phone_evaluation_details[ref_phone] = {"score": round(phone_score, 3), "type": "match"}
            elif entry_type == "substitution":
                # 置換の場合、低めのスコア
                phone_evaluation_details[ref_phone] = {"score": round(base_score * 0.5, 3), "type": "substitution", "recognized_as": hyp_phone}
            elif entry_type == "deletion":
                # 削除の場合、さらに低めのスコア
                phone_evaluation_details[ref_phone] = {"score": round(base_score * 0.2, 3), "type": "deletion"}
            # Insertionは参照音素に対応しないので、個別の参照音素の評価には影響しない

    # 結果をリスト形式で整形
    for phone in reference_phonetic_segments:
        details = phone_evaluation_details.get(phone, {"score": 0.0, "type": "unknown"})
        pronunciation_scores.append({"phone": phone, **details})

    return {
        "phone_pronunciation_scores": pronunciation_scores,
        "overall_pronunciation_score": round(overall_pronunciation_score, 3),
        "levenshtein_distance": int(distance), 
        "levenshtein_path": alignment_path # デバッグ用にパスも含める
    }

def _pinyin_to_phonetic_segments(pinyin_list: List[str]) -> List[str]:
    """
    声調付きピンインのリストを、Kaldiの音素アラインメントと比較可能な
    簡易的な音素セグメントのリストに変換する。
    この関数は非常に単純化されており、実際の中国語音韻論に基づいた知識が必要。後で調べる。
    """
    segments = []
    for pinyin in pinyin_list:
        cleaned_pinyin = re.sub(r'[1-5r]', '', pinyin)
        
        # 最長一致で複合音素を処理
        complex_initials = ['zh', 'ch', 'sh', 'ng']
        complex_finals = [
            'ian', 'iang', 'iong', 'uang', 'ueng', 'uai', 'uan', 'ing', 'ang', 'eng', 'ong',
            'iao', 'uo', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'in', 'un', 'iu', 'ie', 'ui', 'ue', 'er'
        ]
        
        while cleaned_pinyin:
            matched = False
            # 複合子音（声母）のチェック
            for ci in complex_initials:
                if cleaned_pinyin.startswith(ci):
                    segments.append(ci)
                    cleaned_pinyin = cleaned_pinyin[len(ci):]
                    matched = True
                    break
            if matched:
                continue

            # 複合母音（韻母）のチェック
            for cf in complex_finals:
                if cleaned_pinyin.startswith(cf):
                    segments.append(cf)
                    cleaned_pinyin = cleaned_pinyin[len(cf):]
                    matched = True
                    break
            if matched:
                continue
            
            # 残った文字を1文字ずつ処理（単音素として）
            segments.append(cleaned_pinyin[0])
            cleaned_pinyin = cleaned_pinyin[1:]
            
    # 空のセグメントを削除
    segments = [s for s in segments if s]
    return segments


def plot_phoneme_alignment(audio_path: str, alignment_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    音声波形と音素アラインメント情報を可視化する。

    Args:
        audio_path (str): 音声ファイルのパス。
        alignment_data (List[Dict[str, Any]]): 音素アラインメントデータ。
                                             例: [{"phone": "n_B", "start": 0.0, "end": 0.1, "score": 0.95}, ...]
        save_path (Optional[str]): 画像を保存するパス。Noneの場合、画面に表示。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return

    # 図のセットアップ
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='blue')
    plt.title("Waveform with Phoneme Alignment")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # アラインメントの描画
    for entry in alignment_data:
        phoneme = entry.get("phone", "N/A").split('_')[0] # '_B', '_I'などを除去して表示
        start = entry.get("start")
        end = entry.get("end")
        score = entry.get("score", 0.0)

        if start is None or end is None:
            continue

        mid = (start + end) / 2

        # スコアに基づいて色を変える (例: 緑 - 良好, 黄 - 中程度, 赤 - 不良)
        if score >= 0.8:
            color = 'green'
        elif score >= 0.6:
            color = 'gold'
        else:
            color = 'red'

        # 縦線とラベル
        plt.axvline(x=start, color="gray", linestyle="--", linewidth=0.5)
        plt.text(mid, 0.6 * max(y), phoneme, fontsize=10,
                 ha='center', va='bottom', rotation=90,
                 bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7))

    # 最後の音素の終了時刻に縦線
    if alignment_data:
        plt.axvline(x=alignment_data[-1].get("end", 0), color="gray", linestyle="--", linewidth=0.5)
        
    plt.tight_layout()

    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close() # メモリを解放
        print(f"Alignment plot saved to {save_path}")
    else:
        plt.show()

def analyze_chinese_pronunciation(audio_path: str, reference_text: str, pinyin_with_tones: List[str]) -> Dict[str, Any]:
    """
    中国語の発音と声調を別々に分析し、評価結果を返す。

    Args:
        audio_path (str): ユーザーの発音音声ファイルのパス。
        reference_text (str): 参照（正解）となる中国語テキスト。
        pinyin_with_tones (List[str]): 参照テキストに対応する声調付きピンインのリスト。
                                       例: ["ni3", "hao3"]

    Returns:
        Dict[str, Any]: 発音と声調の評価結果。
    Raises:
        PronunciationAnalysisError: 分析中にエラーが発生した場合。
    """
    try:
        # 1. Kaldiによる強制アラインメント
        alignment_result = _run_kaldi_alignment(audio_path, reference_text)
        
        # Kaldiアラインメント結果から音素レベル情報を抽出 (可視化にも使用)
        all_aligned_phones_with_details = []
        for word_data in alignment_result.get("words", []):
            all_aligned_phones_with_details.extend(word_data.get("phones", []))

        # 2. 声調の評価
        tone_evaluation = _evaluate_tone(pinyin_with_tones, alignment_result.get("words", []))

        # 3. 発音（音素レベル）の評価
        reference_phonetic_segments = _pinyin_to_phonetic_segments(pinyin_with_tones)
        pronunciation_evaluation = _evaluate_pronunciation(all_aligned_phones_with_details, reference_phonetic_segments)

        # 4. 音素アラインメントの可視化
        plot_image_path = audio_path.replace(".wav", "_alignment.png")
        plot_phoneme_alignment(audio_path, all_aligned_phones_with_details, save_path=plot_image_path)

        return {
            "overall_score": round((tone_evaluation["overall_tone_score"] + pronunciation_evaluation["overall_pronunciation_score"]) / 2, 3),
            "tone_evaluation": tone_evaluation,
            "pronunciation_evaluation": pronunciation_evaluation,
            "recognized_alignment": alignment_result, # Kaldiの生のアラインメント結果もデバッグ用に含める
            "alignment_plot_path": plot_image_path # 可視化画像のパスを追加
        }
    except KaldiAlignmentError as e:
        raise PronunciationAnalysisError(f"Kaldi alignment failed: {e}")
    except Exception as e:
        raise PronunciationAnalysisError(f"An unexpected error occurred during pronunciation analysis: {e}")

# --- アプリケーション連携のためのメイン関数 ---
def process_user_pronunciation(audio_file_content: bytes, user_text: str, pinyin_for_text: List[str]) -> Dict[str, Any]:
    """
    ユーザーが録音した音声と入力したテキストを受け取り、発音評価を行うメイン関数。
    字数制限のチェックもここで行う。

    Args:
        audio_file_content (bytes): ユーザーが録音した音声ファイルのバイナリデータ。
                                    通常はWAV形式を想定。
        user_text (str): ユーザーが入力した中国語テキスト。
        pinyin_for_text (List[str]): user_text に対応する声調付きピンインのリスト。
                                    例: ["ni3", "hao3"] - これはフロントエンドまたは
                                    別途のピンイン変換サービスで提供されることを想定。

    Returns:
        Dict[str, Any]: 発音と声調の評価結果、またはエラーメッセージ。
    """
    # 1. テキストの字数制限チェック
    if not (MIN_TEXT_LENGTH <= len(user_text) <= MAX_TEXT_LENGTH):
        return {
            "error": "Input text length out of bounds.",
            "min_length": MIN_TEXT_LENGTH,
            "max_length": MAX_TEXT_LENGTH,
            "current_length": len(user_text)
        }

    # 2. 音声ファイルを一時的に保存
    temp_audio_path = "temp_user_audio.wav"
    plot_path = temp_audio_path.replace(".wav", "_alignment.png") # 生成されるプロットのパス
    
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file_content)

        # 3. 発音分析の実行
        evaluation_results = analyze_chinese_pronunciation(temp_audio_path, user_text, pinyin_for_text)
        return evaluation_results
    except PronunciationAnalysisError as e:
        return {"error": f"Pronunciation analysis failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        # 処理後、一時ファイルを削除
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)

## 例：ユーザーインターフェースからの呼び出しをシミュレート

if __name__ == "__main__":
    print("--- Pronunciation Evaluation Simulation ---")

    # シミュレート用のダミー音声ファイル作成 (実際はユーザー録音)
    samplerate = 16000
    duration = 1.0 # seconds
    
    # より複雑な音形を持つダミー音声データ
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    # 複数の周波数で音声を合成し、より「音声」らしくする
    frequency_1 = 440
    frequency_2 = 660
    frequency_3 = 880
    
    # 時間を分けて音声を生成
    # 'n' の部分 (0.0s - 0.1s)
    data_n = 0.5 * np.sin(2. * np.pi * frequency_1 * t[:int(samplerate * 0.1)])
    # 'i' の部分 (0.1s - 0.3s)
    data_i = 0.7 * np.sin(2. * np.pi * frequency_2 * t[int(samplerate * 0.1):int(samplerate * 0.3)])
    # '3' の部分 (0.3s - 0.35s) - 声調を表現するならピッチ変化が必要だが、ここでは単一の周波数で簡易化
    data_tone1 = 0.6 * np.sin(2. * np.pi * frequency_3 * t[int(samplerate * 0.3):int(samplerate * 0.35)])

    # 'h' の部分 (0.35s - 0.45s)
    data_h = 0.4 * np.sin(2. * np.pi * frequency_1 * t[int(samplerate * 0.35):int(samplerate * 0.45)])
    # 'a' の部分 (0.45s - 0.60s)
    data_a = 0.8 * np.sin(2. * np.pi * frequency_2 * t[int(samplerate * 0.45):int(samplerate * 0.60)])
    # 'o' の部分 (0.60s - 0.65s)
    data_o = 0.7 * np.sin(2. * np.pi * frequency_3 * t[int(samplerate * 0.60):int(samplerate * 0.65)])
    # '3' の部分 (0.65s - 0.7s)
    data_tone2 = 0.6 * np.sin(2. * np.pi * frequency_1 * t[int(samplerate * 0.65):int(samplerate * 0.7)])

    # 全体を結合 (間に少し無音を挟むことで音素の区切りを強調)
    audio_segments = [
        data_n, data_i, data_tone1,
        data_h, data_a, data_o, data_tone2
    ]
    audio_data = np.concatenate(audio_segments)
    
    # スケールを調整して16bit WAVに書き込む
    amplitude = np.iinfo(np.int16).max * 0.5
    scaled_audio_data = (audio_data * amplitude).astype(np.int16)

    dummy_audio_file_path = "dummy_user_audio.wav"
    write_wav(dummy_audio_file_path, samplerate, scaled_audio_data)

    with open(dummy_audio_file_path, "rb") as f:
        dummy_audio_content = f.read()

    # ユーザーが入力する中国語テキストと、それに対応するピンイン
    user_input_text_good = "你好"
    pinyin_good = ["ni3", "hao3"]

    user_input_text_long = "这是一段非常长的中文文本，可能超过了 字数制限に引っかかっています。短くしてください" * 2
    user_input_text_short = "文字を入力してください" 
    
    print(f"\n--- Scenario 1: Valid Input ('{user_input_text_good}') ---")
    results_good = process_user_pronunciation(dummy_audio_content, user_input_text_good, pinyin_good)
    print(json.dumps(results_good, indent=2, ensure_ascii=False))

    print(f"\n--- Scenario 2: Text Too Long ({len(user_input_text_long)} chars) ---")
    results_long = process_user_pronunciation(dummy_audio_content, user_input_text_long, []) 
    print(json.dumps(results_long, indent=2, ensure_ascii=False))

    print(f"\n--- Scenario 3: Text Too Short ({len(user_input_text_short)} chars) ---")
    results_short = process_user_pronunciation(dummy_audio_content, user_input_text_short, []) 
    print(json.dumps(results_short, indent=2, ensure_ascii=False))

    # 一時ファイルを削除
    if os.path.exists(dummy_audio_file_path):
        os.remove(dummy_audio_file_path)
    # 生成されたプロット画像も削除（もし残っていれば）
    plot_path_good = dummy_audio_file_path.replace(".wav", "_alignment.png")
    if os.path.exists(plot_path_good):
        os.remove(plot_path_good)