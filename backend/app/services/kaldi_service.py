import subprocess
import os
import json
import re
import Levenshtein # Levenshtein距離計算のため
import numpy as np # DPテーブルのため
from typing import Dict, Any, List, Optional, Set, Tuple
import librosa
import librosa.display
import matplotlib.pyplot as plt
import uuid # 一時ファイル名の一意性のため
import logging # ログ出力のため

# --- ロギングの設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 環境変数の設定 ---
KALDI_ROOT = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi")
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
    音素レベルの情報（発音時間、スコアなど）を取得します。

    Args:
        audio_path (str): 処理する音声ファイルのパス。
        text (str): 音声に対応する参照テキスト（中国語）。

    Returns:
        Dict[str, Any]: アラインメント結果を含む辞書。
                        例: {"words": [{"word": "你好", "start": 0.1, "end": 0.5, "phones": [...]}]}
    Raises:
        KaldiAlignmentError: Kaldiの実行に失敗した場合。
    """

def _run_kaldi_alignment(audio_path: str, text: str) -> Dict[str, Any]:
    """
    Kaldi を使用して音声ファイルとテキストの強制アラインメントを実行し、
    音素レベルの情報（発音時間、スコアなど）を取得します。

    Args:
        audio_path (str): 処理する音声ファイルのパス。
        text (str): 音声に対応する参照テキスト（中国語）。

    Returns:
        Dict[str, Any]: アラインメント結果を含む辞書。
                        例: {"words": [{"word": "你好", "start": 0.1, "end": 0.5, "phones": [...]}]}
    Raises:
        KaldiAlignmentError: Kaldiの実行に失敗した場合。
    """

    logger.info(f"Running Kaldi alignment for: {audio_path} with text: {text}")
    try:
        command = [
            "python", "serenakurashina_kaldi_aligner_wrapper.py",  # 必要に応じてパスを修正
            "--audio", audio_path,
            "--text", text,
            "--model", MANDARIN_MODEL_PATH
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        alignment_data = json.loads(result.stdout)
        return alignment_data
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaldi alignment failed: {e.stderr}")
        raise KaldiAlignmentError(f"Kaldi alignment process failed: {e.stderr.strip()}")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Kaldi alignment output as JSON: {result.stdout}")
        raise KaldiAlignmentError("Kaldi alignment output was not valid JSON.")

# --- 評価関数 ---
def _evaluate_tone(pinyin_with_tones: List[str], aligned_words_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    声調の評価を行います。
    アラインメント結果から声調の確信度や逸脱を推定します。
    Kaldiの出力が声調に関する情報（例: tone GMMスコア）を含む場合に有効です。

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
            # 声調音素があるかチェック (例: '1_T', '2_T'...)
            has_tone_phone = any(re.fullmatch(r'[1-5]_T', phone_data.get("phone", "")) for phone_data in aligned_words_data[i]["phones"])

            if has_tone_phone:
                relevant_tone_phones_scores = [
                    phone_data["score"] for phone_data in aligned_words_data[i]["phones"]
                    if re.fullmatch(r'[1-5]_T', phone_data.get("phone", ""))
                ]
                if relevant_tone_phones_scores:
                    avg_tone_score_from_kaldi = sum(relevant_tone_phones_scores) / len(relevant_tone_phones_scores)
                    # Kaldiのスコアを線形的に0から1の範囲にマッピングする
                    # 0.7を基準として、スコアが高いほど良いと仮定
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

def _levenshtein_alignment_with_dp(ref: List[str], hyp: List[str]) -> Tuple[int, np.ndarray]:
    """
    Levenshtein距離とDPテーブルを計算します。
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
            dp[i][j] = min(dp[i-1][j] + 1,      # 削除 (Deletion)
                           dp[i][j-1] + 1,      # 挿入 (Insertion)
                           dp[i-1][j-1] + cost) # 置換 (Substitution) or 一致 (Match)

    return dp[n][m], dp  # 距離とDPテーブル

def _traceback_levenshtein_path(dp_table: np.ndarray, ref: List[str], hyp: List[str]) -> List[Dict[str, Any]]:
    """
    Levenshtein DPテーブルをトレースバックしてアラインメントパスを生成します。
    Args:
        dp_table (np.ndarray): DPテーブル
        ref (List[str]): 参照音素列
        hyp (List[str]): 仮説音素列
    Returns:
        List[Dict[str, Any]]: アラインメントパスのリスト。
                               例: [{"type": "match", "ref_phone": "n", "hyp_phone": "n"}]
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
            # 不正な状態のエラーハンドリング
            logger.error("Levenshtein traceback error: No valid path found.")
            raise PronunciationAnalysisError("Levenshtein traceback error: No valid path found.")

    return path[::-1] # パスを逆順にする

# --- ピンイン音素分解 ---
class PinyinToPhoneticConverter:
    """
    本土マンダリンの全ピンイン音節に対応
    """

    def __init__(self):
        # 声母（Initial consonants）- 21個
        self.initials = {
            # 無声子音
            'b', 'p', 'm', 'f',           # 唇音
            'd', 't', 'n', 'l',           # 舌尖音
            'g', 'k', 'h',                # 舌根音
            'j', 'q', 'x',                # 舌面音
            'zh', 'ch', 'sh', 'r',        # 舌尖後音（巻舌音）
            'z', 'c', 's'                 # 舌尖前音
        }

        # 韻母（Finals）- 39個の基本韻母
        self.finals = {
            # 単韻母
            'a', 'o', 'e', 'i', 'u', 'ü',
            # 複韻母
            'ai', 'ei', 'ao', 'ou',
            'ia', 'ie', 'iao', 'iu', 'iou',
            'ua', 'uo', 'uai', 'ui', 'uei',
            'üe',
            # 鼻韻母
            'an', 'en', 'ang', 'eng', 'ong',
            'ian', 'in', 'iang', 'ing', 'iong',
            'uan', 'un', 'uang', 'ueng',
            'üan', 'ün',
            # 特殊韻母
            'er'  # 児化音
        }

        # 単独韻母（声母なしで使用可能）
        self.standalone_finals = {
            'a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er'
        }

        # y/w で始まる特殊形
        self.y_w_variants = {
            'yi': 'i', 'ya': 'ia', 'yao': 'iao', 'you': 'iou', 'yan': 'ian',
            'yin': 'in', 'yang': 'iang', 'ying': 'ing', 'yong': 'iong', 'ye': 'ie',
            'wu': 'u', 'wa': 'ua', 'wo': 'uo', 'wai': 'uai', 'wei': 'uei',
            'wan': 'uan', 'wen': 'un', 'wang': 'uang', 'weng': 'ueng',
            'yu': 'ü', 'yue': 'üe', 'yuan': 'üan', 'yun': 'ün'
        }

        # 全ピンイン音節の完全リスト（412個の標準音節）
        self.valid_syllables = self._generate_valid_syllables()

        # 音素分解ルール
        self.phoneme_rules = self._create_phoneme_rules()

    def _generate_valid_syllables(self) -> Set[str]:
        """標準中国語の全有効音節を生成"""
        valid = set()

        # 声母+韻母の組み合わせ
        for initial in self.initials:
            for final in self.finals:
                if self._is_valid_combination(initial, final):
                    valid.add(initial + final)

        # 単独韻母
        for final in self.standalone_finals:
            valid.add(final)

        # y/w変形
        valid.update(self.y_w_variants.keys())

        return valid

    def _is_valid_combination(self, initial: str, final: str) -> bool:
        """声母と韻母の組み合わせが有効かチェック"""
        # 基本的な制約ルール
        palatals = {'j', 'q', 'x'}  # 舌面音
        retroflexes = {'zh', 'ch', 'sh', 'r'}  # 巻舌音
        sibilants = {'z', 'c', 's'}  # 舌尖前音

        # j, q, x は i, ü で始まる韻母としか結合しない
        if initial in palatals:
            return final.startswith(('i', 'ü'))

        # zh, ch, sh, r は i, ü で始まる韻母と結合しない（iを除く）
        if initial in retroflexes:
            if final.startswith('ü'):
                return False
            if final.startswith('i') and final != 'i':
                return False

        # z, c, s は i, ü で始まる韻母と結合しない（iを除く）
        if initial in sibilants:
            if final.startswith('ü'):
                return False
            if final.startswith('i') and final != 'i':
                return False

        # その他の制約
        if final == 'ong' and initial in {'j', 'q', 'x', 'z', 'c', 's'}:
            return False

        return True

    def _create_phoneme_rules(self) -> Dict[str, List[str]]:
        """各音節の音素分解ルールを定義"""
        rules = {}

        # 基本的な分解パターン
        for initial in self.initials:
            for final in self.finals:
                syllable = initial + final
                if syllable in self.valid_syllables:
                    rules[syllable] = self._decompose_syllable(initial, final)

        # 単独韻母
        for final in self.standalone_finals:
            if final in self.valid_syllables:
                rules[final] = self._decompose_final(final)

        # y/w変形
        for variant, original in self.y_w_variants.items():
            rules[variant] = self._decompose_final(original)

        return rules

    def _decompose_syllable(self, initial: str, final: str) -> List[str]:
        """声母と韻母を個別の音素に分解"""
        phonemes = []

        # 声母の分解
        if len(initial) == 2:  # zh, ch, sh等
            phonemes.extend(list(initial))
        else:
            phonemes.append(initial)

        # 韻母の分解
        phonemes.extend(self._decompose_final(final))

        return phonemes

    def _decompose_final(self, final: str) -> List[str]:
        """韻母を個別の音素に分解"""
        # 複雑な韻母から順にチェック
        complex_finals = {
            'iang': ['i', 'a', 'ng'],
            'iong': ['i', 'o', 'ng'],
            'uang': ['u', 'a', 'ng'],
            'ueng': ['u', 'e', 'ng'],
            'üan': ['ü', 'a', 'n'],
            'ian': ['i', 'a', 'n'],
            'iao': ['i', 'a', 'o'],
            'iou': ['i', 'o', 'u'],
            'uai': ['u', 'a', 'i'],
            'uei': ['u', 'e', 'i'],
            'uan': ['u', 'a', 'n'],
            'üe': ['ü', 'e'],
            'ang': ['a', 'ng'],
            'eng': ['e', 'ng'],
            'ing': ['i', 'ng'],
            'ong': ['o', 'ng'],
            'ai': ['a', 'i'],
            'ei': ['e', 'i'],
            'ao': ['a', 'o'],
            'ou': ['o', 'u'],
            'an': ['a', 'n'],
            'en': ['e', 'n'],
            'in': ['i', 'n'],
            'un': ['u', 'n'],
            'ün': ['ü', 'n'],
            'ia': ['i', 'a'],
            'ie': ['i', 'e'],
            'iu': ['i', 'u'],
            'ua': ['u', 'a'],
            'uo': ['u', 'o'],
            'ui': ['u', 'i'],
            'er': ['e', 'r'],
        }

        if final in complex_finals:
            return complex_finals[final]
        else:
            # 単一音素
            return [final]

    def pinyin_to_phonetic_segments(self, pinyin_list: List[str]) -> List[str]:
        """
        ピンインリストを音素セグメントに変換

        Args:
            pinyin_list: 声調付きピンインのリスト ['ni3', 'hao3']

        Returns:
            音素セグメントのリスト ['n', 'i', 'h', 'a', 'o']
        """
        all_segments = []

        for pinyin in pinyin_list:
            # 声調番号を除去
            clean_pinyin = re.sub(r'[0-5]', '', pinyin.lower().strip())

            # 空文字列をスキップ
            if not clean_pinyin:
                continue

            # 'r'化音の処理
            if clean_pinyin.endswith('r') and clean_pinyin != 'er':
                base_syllable = clean_pinyin[:-1]
                if base_syllable in self.phoneme_rules:
                    segments = self.phoneme_rules[base_syllable].copy()
                    segments.append('r')
                    all_segments.extend(segments)
                    continue

            # 通常の音節処理
            if clean_pinyin in self.phoneme_rules:
                all_segments.extend(self.phoneme_rules[clean_pinyin])
            else:
                logger.warning(f"Unknown syllable '{clean_pinyin}', using character-by-character fallback")
                all_segments.extend(list(clean_pinyin))

        return all_segments

    def get_syllable_info(self, pinyin: str) -> Dict[str, Any]:
        """音節の詳細情報を取得（デバッグ用）"""
        clean_pinyin = re.sub(r'[0-5]', '', pinyin.lower().strip())

        if clean_pinyin not in self.valid_syllables:
            return {"valid": False, "error": f"Invalid syllable: {clean_pinyin}"}

        # 声母と韻母を分析
        initial = ""
        final = ""

        # 長い声母から順にチェック
        for init in sorted(self.initials, key=len, reverse=True):
            if clean_pinyin.startswith(init):
                initial = init
                final = clean_pinyin[len(init):]
                break

        if not initial:  # 単独韻母またはy/w変形
            if clean_pinyin in self.y_w_variants:
                final = self.y_w_variants[clean_pinyin]
            else:
                final = clean_pinyin

        return {
            "valid": True,
            "original": pinyin,
            "clean": clean_pinyin,
            "initial": initial,
            "final": final,
            "phonemes": self.phoneme_rules.get(clean_pinyin, [])
        }

def _evaluate_pronunciation(aligned_phones_details: List[Dict[str, Any]], reference_phonetic_segments: List[str]) -> Dict[str, Any]:
    """
    発音（音素レベル）の評価を行います。
    Levenshtein距離を使用して、参照とアラインされた音素列の間の距離を測定し、スコア化します。

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
    # 各参照音素に対応する評価を初期化（デフォルトは全体スコア、タイプは"unmatched"）
    # 複数回現れる音素に対応するため、辞書ではなくリストのリストを使用
    phone_evaluation_details: Dict[str, List[Dict[str, Any]]] = {phone: [] for phone in reference_phonetic_segments}
    # 参照音素のインデックスを保持
    ref_phone_indices: Dict[str, List[int]] = {}
    for idx, phone in enumerate(reference_phonetic_segments):
        ref_phone_indices.setdefault(phone, []).append(idx)

    # アラインメントパスに基づいて評価を更新
    # ここでは、簡略化のため、パスの各エントリが参照音素の評価に直接寄与すると仮定します。
    # より高度な評価では、音素の重複や、特定の音素が複数回発音された場合の扱いを考慮する必要があります。
    for path_entry in alignment_path:
        ref_phone = path_entry.get("ref_phone")
        hyp_phone = path_entry.get("hyp_phone")
        entry_type = path_entry["type"]

        if ref_phone: # 参照音素がある場合のみ評価対象
            details = {"type": entry_type}
            if entry_type == "match":
                # マッチした場合、Kaldiの個別スコアと全体スコアを重み付け
                # hyp_phone_idxを正確に特定するために、パスから逆算する必要があるが、
                # ここでは簡略化のため、hyp_phoneの最初の出現スコアを仮定
                try:
                    hyp_phone_idx = aligned_pure_phones.index(hyp_phone)
                    kaldi_raw_score = aligned_phone_raw_scores[hyp_phone_idx]
                except ValueError:
                    kaldi_raw_score = 0.0 # 見つからない場合は0
                phone_score = (0.7 * kaldi_raw_score) + (0.3 * overall_pronunciation_score)
                details["score"] = round(phone_score, 3)
            elif entry_type == "substitution":
                details["score"] = round(overall_pronunciation_score * 0.5, 3)
                details["recognized_as"] = hyp_phone
            elif entry_type == "deletion":
                details["score"] = round(overall_pronunciation_score * 0.2, 3)
            else: # Insertionは参照音素に対応しない
                details["score"] = 0.0 # 念のため

            # 参照音素の評価リストに追加
            # これは、同じ参照音素が複数回現れる場合にすべての評価を保持するためです。
            # 最終的なスコアは、これらの評価を平均するなどして算出できます。
            phone_evaluation_details[ref_phone].append(details)


    # 結果をリスト形式で整形し、最終的なスコアを計算
    final_pronunciation_scores = []
    for phone in reference_phonetic_segments:
        evals = phone_evaluation_details.get(phone, [])
        if evals:
            # 複数の評価がある場合は、最も良いスコアを採用するか、平均を取るかなど、ポリシーを決定
            # ここでは、最も良いスコアを採用（よりポジティブなフィードバックを与えるため）
            best_score = max(e.get("score", 0.0) for e in evals)
            # 各評価タイプを列挙
            types = list(set(e["type"] for e in evals))
            recognized_as = list(set(e["recognized_as"] for e in evals if "recognized_as" in e))
            final_pronunciation_scores.append({
                "phone": phone,
                "score": round(best_score, 3),
                "types": types, # 複数のタイプがある可能性
                "recognized_as": recognized_as if recognized_as else None
            })
        else:
            final_pronunciation_scores.append({"phone": phone, "score": 0.0, "types": ["unmatched"]})

    return {
        "phone_pronunciation_scores": final_pronunciation_scores,
        "overall_pronunciation_score": round(overall_pronunciation_score, 3),
        "levenshtein_distance": int(distance),
        "levenshtein_path": alignment_path # デバッグ用にパスも含める
    }

# 音素レベルの評価
def _pinyin_to_phonetic_segments(pinyin_list: List[str]) -> List[str]:
    converter = PinyinToPhoneticConverter()
    return converter.pinyin_to_phonetic_segments(pinyin_list)

def plot_phoneme_alignment(audio_path: str, alignment_data: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    音声波形と音素アラインメント情報を可視化します。

    Args:
        audio_path (str): 音声ファイルのパス。
        alignment_data (List[Dict[str, Any]]): 音素アラインメントデータ。
                                                例: [{"phone": "n_B", "start": 0.0, "end": 0.1, "score": 0.95}, ...]
        save_path (Optional[str]): 画像を保存するパス。Noneの場合、画面に表示。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return

    # 図のセットアップ
    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='blue', ax=ax)
    ax.set_title("Waveform with Phoneme Alignment")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

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
        ax.axvline(x=start, color="gray", linestyle="--", linewidth=0.5)
        ax.text(mid, 0.6 * np.max(np.abs(y)), phoneme, fontsize=10, # y軸の範囲を考慮してテキスト位置を調整
                 ha='center', va='bottom', rotation=90,
                 bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7))

    # 最後の音素の終了時刻に縦線
    if alignment_data:
        # alignment_dataがソートされていることを前提に、最後の要素の'end'を使用
        last_end_time = max(entry.get("end", 0) for entry in alignment_data)
        ax.axvline(x=last_end_time, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # 特定のfigオブジェクトを閉じる
        logger.info(f"Alignment plot saved to {save_path}")
    else:
        plt.show()
        plt.close(fig) # show()の後も閉じる

def analyze_chinese_pronunciation(audio_path: str, reference_text: str, pinyin_with_tones: List[str]) -> Dict[str, Any]:
    """
    中国語の発音と声調を別々に分析し、評価結果を返します。

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
        logger.exception(f"Kaldi alignment failed for {audio_path}.") # スタックトレースも出力
        raise PronunciationAnalysisError(f"Kaldi alignment failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during pronunciation analysis for {audio_path}.") # スタックトレースも出力
        raise PronunciationAnalysisError(f"An unexpected error occurred during pronunciation analysis: {e}")

# --- アプリケーション連携のためのメイン関数 ---
def process_user_pronunciation(audio_file_content: bytes, user_text: str, pinyin_for_text: List[str]) -> Dict[str, Any]:
    """
    ユーザーが録音した音声と入力したテキストを受け取り、発音評価を行うメイン関数。
    字数制限のチェックもここで行います。

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
    # 一意なファイル名を生成
    unique_id = uuid.uuid4()
    temp_audio_path = f"temp_user_audio_{unique_id}.wav"
    plot_path = f"temp_user_audio_{unique_id}_alignment.png"

    try:
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file_content)

        # 3. 発音分析の実行
        evaluation_results = analyze_chinese_pronunciation(temp_audio_path, user_text, pinyin_for_text)
        # 生成されたプロットファイルのパスを結果に含める
        evaluation_results["alignment_plot_path"] = plot_path
        return evaluation_results
    except PronunciationAnalysisError as e:
        return {"error": f"Pronunciation analysis failed: {e}"}
    except Exception as e:
        logger.exception("An unexpected error occurred in process_user_pronunciation.")
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        # 処理後、一時ファイルを削除
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Removed temporary audio file: {temp_audio_path}")
        if os.path.exists(plot_path):
            os.remove(plot_path)
            logger.info(f"Removed temporary plot file: {plot_path}")