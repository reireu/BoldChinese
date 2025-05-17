import subprocess
import os
import difflib

def run_kaldi(audio_path: str) -> str:
    """
    Kaldi を実行し、音声ファイルからテキストを抽出する。
    ※ Kaldi のセットアップに応じてこのコマンドは変更してください。
    """
    try:
        # 例: Kaldi のスクリプトを呼び出して音声ファイルを処理
        result = subprocess.run(
            ["bash", "run_kaldi.sh", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # Kaldi の出力（テキスト）を取得
        recognized_text = result.stdout.decode("utf-8").strip()
        return recognized_text

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Kaldi failed: {e.stderr.decode('utf-8')}")

def calculate_similarity(reference: str, hypothesis: str) -> float:
    """
    参考テキスト（正解）と Kaldi の出力の類似度（0〜1）を計算。
    Levenshtein などに置き換え可能。
    """
    matcher = difflib.SequenceMatcher(None, reference.lower(), hypothesis.lower())
    return round(matcher.ratio(), 3)

def analyze_pronunciation(audio_path: str, reference_text: str) -> dict:
    """
    Kaldi 実行＋スコア計算して結果を返す。
    """
    recognized_text = run_kaldi(audio_path)
    score = calculate_similarity(reference_text, recognized_text)

    return {
        "score": score,
        "reference": reference_text,
        "recognized": recognized_text
    }
