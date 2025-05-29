from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from app.services.kaldi_service import analyze_chinese_pronunciation, PronunciationAnalysisError
import shutil
import tempfile
import os
import json # pypinyinの代わりにフロントエンドからpinyinを受け取る場合のためにimport
from pypinyin import pinyin, Style # 中国語テキストからピンインを生成するため
from typing import Optional

router = APIRouter()

# 文字数の制限(ここを変えたらkaidi_service.pyの方も変えるのを忘れずに)
MAX_TEXT_LENGTH = 600
MIN_TEXT_LENGTH = 1

@router.post("/analyze")
async def analyze(
    audio: UploadFile = File(..., description="ユーザーが録音したWAV形式の音声ファイル"),
    text: str = Form(..., max_length=MAX_TEXT_LENGTH, min_length=MIN_TEXT_LENGTH, description="発音評価の基準となる中国語のテキスト")
):
    """
    ユーザーの発音と声調を、参照テキストに基づいて分析します。

    引数:
        audio (UploadFile): ユーザーが録音した音声ファイル（WAV形式を想定）
        text (str): 発音分析の基準となる中国語の参照テキスト

    戻り値:
        dict: 以下の情報を含む辞書を返します：
              - 全体スコア
              - 声調評価
              - 発音評価
              - 生のアライメント結果
              - アライメント画像のパス

    例外:
        HTTPException: 入力の検証に失敗した場合、または分析中にエラーが発生した場合
    """

    audio_path: Optional[str] = None
    plot_path: Optional[str] = None

    try:
        # 2. 音声ファイルを一時ファイルとして保存
        # tempfile.NamedTemporaryFile は close() 時に自動で削除されるが、
        # 明示的な削除を行うため delete=False を指定して使用
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            audio_path = tmp.name
        
        # 3. 中国語テキストからピンインを生成（バックエンドで生成する方式を採用）
        try:
            # pypinyin を使ってピンインを生成（声調付きスタイルを指定）
            # [['nǐ'], ['hǎo']] のような形式が返るため、['ni3', 'hao3'] に変換
            pinyin_raw = pinyin(text, style=Style.TONE)
            pinyin_for_text = [item[0] for item in pinyin_raw]
            
            # ピンインリストが空になるケース（無効な中国語など）を検出してエラーとする
            if not pinyin_for_text:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="中国語テキストからピンインを生成できませんでした。入力内容をご確認ください。"
                )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"中国語テキストからピンインの生成に失敗しました: {str(e)}"
            )

        # 4. 発音分析の実行
        # kaldi_service.py に定義された analyze_chinese_pronunciation 関数を呼び出す
        results = analyze_chinese_pronunciation(audio_path, text, pinyin_for_text)
        
        # 生成されたアライメント画像のパスを保存
        plot_path = results.get("alignment_plot_path")

        return results

    except PronunciationAnalysisError as e:
        # kaldi_service.py からスローされるカスタム例外を捕捉
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # クライアント起因のエラー
            detail=f"発音分析に失敗しました: {e}"
        )
    except HTTPException:
        # すでに明示的にスローされた HTTPException はそのまま再スロー
        raise
    except Exception as e:
        # それ以外の予期しないエラーを捕捉
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"予期せぬサーバーエラーが発生しました: {str(e)}"
        )
    finally:
        # 一時ファイルと生成されたプロット画像をクリーンアップ
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"一時音声ファイルを削除しました: {audio_path}")
        if plot_path and os.path.exists(plot_path):
            os.remove(plot_path)
            print(f"アライメント画像ファイルを削除しました: {plot_path}")
