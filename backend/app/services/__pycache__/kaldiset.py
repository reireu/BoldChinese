#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path

def print_separator(title):
    print("\n============================================================")
    print(f" {title}")
    print("============================================================\n")

def check_environment():
    kaldi_root = os.getenv("KALDI_ROOT")
    if not kaldi_root:
        print("❌ KALDI_ROOT環境変数が設定されていません")
        return False
    
    required_files = {
        "path.sh": Path(kaldi_root) / "path.sh",
        "env.sh": Path(kaldi_root) / "tools/env.sh"
    }
    
    for name, path in required_files.items():
        if not path.exists():
            print(f"❌ {name}が見つかりません: {path}")
            return False
    return True

def check_binaries():
    binaries = {
        "compute-mfcc-feats": ["src/bin/compute-mfcc-feats", "src/featbin/compute-mfcc-feats"],
        "gmm-align": ["src/gmmbin/gmm-align"],
        "copy-feats": ["src/bin/copy-feats"],
        "fstinfo": ["tools/openfst/bin/fstinfo"]
    }
    
    print_separator("重要バイナリの存在確認")
    
    for binary, paths in binaries.items():
        found = False
        for path in paths:
            if shutil.which(binary) or Path(os.getenv("KALDI_ROOT", "")) / path:
                print(f"✅ {path}")
                found = True
                break
        if not found:
            print(f"❌ バイナリが見つかりません: {binary}")

def print_next_steps():
    print_separator("次のステップ")
    print("1. path.shが正しくソースされているか確認してください")
    print("2. PATHに正しいディレクトリが含まれているか確認してください")
    print("3. 必要な音響モデルファイルが存在するか確認してください")

def main():
    print_separator("Kaldi環境チェック開始")
    
    if not check_environment():
        print("\n❌ 基本的な環境設定に問題があります")
        print_next_steps()
        return 1
    
    check_binaries()
    print_next_steps()
    
    print_separator("完了")
    return 0

def setup_kaldi_environment():
    try:
        kaldi_root = Path("/Users/serenakurashina/kaldi")
        
        # 必要なパスの確認
        if not kaldi_root.exists():
            print(f"エラー: Kaldiルートディレクトリが存在しません: {kaldi_root}")
            return 1

        # シェル設定ファイルの生成
        shell_config = f"""# Kaldi環境設定
export KALDI_ROOT={kaldi_root}
if [ -f $KALDI_ROOT/tools/env.sh ]; then
    source $KALDI_ROOT/tools/env.sh
fi
if [ -f $KALDI_ROOT/path.sh ]; then
    source $KALDI_ROOT/path.sh
fi
export PATH=$KALDI_ROOT/src/bin:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/gmmbin:$KALDI_ROOT/tools/openfst/bin:$PATH
"""
        # 設定ファイルの保存
        kaldi_env_path = Path.home() / ".kaldi_env"
        with open(kaldi_env_path, "w") as f:
            f.write(shell_config)

        # .zshrcへの追記確認
        zshrc_path = Path.home() / ".zshrc"
        with open(zshrc_path, "r") as f:
            if "source ~/.kaldi_env" not in f.read():
                with open(zshrc_path, "a") as f:
                    f.write("\n# Kaldi環境設定\nsource ~/.kaldi_env\n")

        print("✅ Kaldi環境設定が完了しました")
        print("ターミナルを再起動するか、以下のコマンドを実行してください:")
        print("source ~/.zshrc")
        return 0

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(setup_kaldi_environment())