import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Any

# kaldiのルートディレクトリとモデルのパスを環境変数から取得
KALDI_ROOT = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi")
MODEL_ROOT = os.getenv("MANDARIN_MODEL_PATH", f"{KALDI_ROOT}/egs/mandarin_bn_bci")

def prepare_audio(audio_path: str, out_dir: str) -> str:
    """Convert audio to Kaldi format"""
    #  (16kHz, mono, WAV)
    out_path = os.path.join(out_dir, "audio.wav")
    cmd = [
        "sox", audio_path,
        "-r", "16000",
        "-c", "1",
        "-b", "16",
        out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path

def prepare_text(text: str, out_dir: str) -> str:
    """Prepare text for Kaldi"""
    text_path = os.path.join(out_dir, "text")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"utt1 {text}\n")
    return text_path

def run_alignment(audio_path: str, text: str, model_dir: str) -> Dict[str, Any]:
    """Run forced alignment using Kaldi"""
    
    # ディレクトリ作成
    work_dir = "tmp_align"
    os.makedirs(work_dir, exist_ok=True)

    try:
        #ファイルの準備
        wav_path = prepare_audio(audio_path, work_dir)
        text_path = prepare_text(text, work_dir)

        # wav.scpの作成
        with open(os.path.join(work_dir, "wav.scp"), "w") as f:
            f.write(f"utt1 {wav_path}\n")

        # Run Kaldi alignment
        align_cmd = [
            "bash", 
            os.path.join(model_dir, "align.sh"),
            work_dir,
            "utt1"
        ]
        
        result = subprocess.run(
            align_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse alignment output
        align_output = os.path.join(work_dir, "align.json")
        with open(align_output) as f:
            alignment = json.load(f)

        return alignment

    finally:
        # Cleanup
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

def main():
    parser = argparse.ArgumentParser(description="Kaldi forced alignment wrapper")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--text", required=True, help="Input text to align")
    parser.add_argument("--model", required=True, help="Path to Kaldi model directory")
    
    args = parser.parse_args()

    try:
        result = run_alignment(args.audio, args.text, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()