import os
import sys
import shutil
from pathlib import Path

# Kaldiルートディレクトリのパス
KALDI_ROOT = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi")
# WSJ S5の例のルートディレクトリ (align.shがここからの相対パスを期待するため)
WSJ_S5_ROOT = os.getenv("WSJ_S5_ROOT", "/Users/serenakurashina/kaldi/egs/wsj/s5")
# align.sh のWSJ_S5_ROOTからの相対パス
ALIGN_SCRIPT_RELATIVE_PATH = "steps/online/nnet2/align.sh" 

def check_common_path_sh() -> bool:
    """common_path.shの存在確認とパスの検証"""
    common_path_locations = [
        Path(KALDI_ROOT) / "tools" / "config" / "common_path.sh",
        Path(WSJ_S5_ROOT) / "tools" / "config" / "common_path.sh"
    ]
    
    for path in common_path_locations:
        if path.exists():
            print(f"DEBUG: common_path.sh found at: {path}", file=sys.stderr)
            # ファイルの内容を確認
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    print(f"DEBUG: common_path.sh content:\n{content}", file=sys.stderr)
                return True
            except Exception as e:
                print(f"ERROR: common_path.shの読み込みに失敗: {e}", file=sys.stderr)
    
    print("WARNING: common_path.shが見つかりません", file=sys.stderr)
    return False

def check_kaldi_environment() -> bool:
    """Kaldi環境の確認（WSJ_S5_ROOT基準に修正）"""
    print("DEBUG: check_kaldi_environment() 開始", file=sys.stderr)
    
    # common_path.shのチェックを追加
    if not check_common_path_sh():
        print("WARNING: common_path.shの確認に失敗しました", file=sys.stderr)
    
if __name__ == "__main__":
    # スクリプトの実行
    if check_kaldi_environment():
        print("DEBUG: Kaldi環境チェック成功", file=sys.stderr)
    else:
        print("DEBUG: Kaldi環境チェック失敗", file=sys.stderr)