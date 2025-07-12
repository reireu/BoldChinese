#!/usr/bin/env python3
"""
Kaldiのビルド状況を確認し、必要に応じてビルドを実行するスクリプト
"""

import subprocess
import sys
from pathlib import Path
import os

KALDI_ROOT = "/Users/serenakurashina/kaldi"

def print_separator(title):
    """デバッグ出力の区切り線"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_directory_contents(directory, description):
    """ディレクトリの内容を確認"""
    path = Path(directory)
    if not path.exists():
        print(f"❌ {description} が存在しません: {path}")
        return False
    
    files = list(path.iterdir())
    print(f"📁 {description}: {path}")
    print(f"   ファイル数: {len(files)}")
    
    if len(files) > 0:
        print("   内容（最初の10件）:")
        for file in files[:10]:
            if file.is_file():
                print(f"   📄 {file.name}")
            elif file.is_dir():
                print(f"   📁 {file.name}/")
        if len(files) > 10:
            print(f"   ... 他 {len(files) - 10} 件")
    else:
        print("   ⚠️ ディレクトリが空です")
    
    return True

def run_command_with_output(cmd, description, cwd=None, timeout=300):
    """コマンドを実行して結果を表示"""
    print(f"\n🔍 {description}")
    print(f"コマンド: {cmd}")
    print(f"実行ディレクトリ: {cwd or os.getcwd()}")
    print("実行中...")
    
    try:
        # リアルタイムで出力を表示するため、Popenを使用
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=cwd
        )
        
        # リアルタイム出力
        for line in process.stdout:
            print(f"  {line.rstrip()}")
        
        process.wait(timeout=timeout)
        
        print(f"終了コード: {process.returncode}")
        return process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ タイムアウト ({timeout}秒)")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def main():
    print_separator("Kaldiビルド状況確認")
    
    # 1. Kaldiのsrcディレクトリ構造確認
    print_separator("Kaldiソースディレクトリ確認")
    
    src_path = Path(KALDI_ROOT) / "src"
    if not src_path.exists():
        print(f"❌ Kaldiのsrcディレクトリが存在しません: {src_path}")
        print("Kaldiが正しくクローン/ダウンロードされていない可能性があります。")
        return
    
    check_directory_contents(src_path, "src")
    
    # 重要なbinディレクトリの確認
    important_bins = ["bin", "featbin", "gmmbin", "fstbin"]
    missing_bins = []
    
    for bin_name in important_bins:
        bin_path = src_path / bin_name
        if check_directory_contents(bin_path, f"src/{bin_name}"):
            # バイナリファイルがあるかチェック
            binaries = [f for f in bin_path.iterdir() if f.is_file() and not f.name.endswith('.cc') and not f.name.endswith('.h')]
            if len(binaries) == 0:
                print(f"   ⚠️ バイナリファイルが見つかりません（コンパイルが必要）")
                missing_bins.append(bin_name)
            else:
                print(f"   ✅ バイナリファイルを確認: {len(binaries)} 個")
        else:
            missing_bins.append(bin_name)
    
    # 2. 具体的なバイナリ存在確認
    print_separator("重要バイナリの存在確認")
    
    important_binaries = [
        "src/bin/compute-mfcc-feats",
        "src/featbin/compute-mfcc-feats",  # compute-mfcc-featsは実際にはfeatbinにある
        "src/gmmbin/gmm-align",
        "src/bin/copy-feats",
        "tools/openfst/bin/fstinfo",
    ]
    
    existing_binaries = []
    for binary_path in important_binaries:
        full_path = Path(KALDI_ROOT) / binary_path
        if full_path.exists():
            print(f"✅ {binary_path}")
            existing_binaries.append(binary_path)
        else:
            print(f"❌ {binary_path}")
    
    # 3. ビルド状況の判定
    print_separator("ビルド状況判定")
    
    if len(existing_binaries) == 0:
        print("❌ Kaldiがまったくビルドされていません。")
        print("以下の手順でKaldiをビルドする必要があります：")
        build_needed = True
    elif len(existing_binaries) < len(important_binaries) / 2:
        print("⚠️ Kaldiが部分的にしかビルドされていません。")
        print("再ビルドを推奨します。")
        build_needed = True
    else:
        print("✅ Kaldiは概ねビルド済みのようです。")
        print("ただし、一部のバイナリが不足している可能性があります。")
        build_needed = False
    
    # 4. ビルドの実行提案
    if build_needed:
        print_separator("Kaldiビルド実行")
        
        print("Kaldiをビルドしますか？")
        print("これには時間がかかる場合があります（30分〜数時間）。")
        
        response = input("ビルドを実行しますか？ (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("\n📦 Kaldiのビルドを開始します...")
            
            # toolsのビルド
            print_separator("Step 1: tools のビルド")
            tools_success = run_command_with_output(
                "make -j$(nproc 2>/dev/null || echo 4)", 
                "toolsのビルド",
                cwd=Path(KALDI_ROOT) / "tools"
            )
            
            if not tools_success:
                print("❌ toolsのビルドに失敗しました。")
                print("手動で以下を実行してください：")
                print(f"cd {KALDI_ROOT}/tools && make")
                return
            
            # srcのビルド
            print_separator("Step 2: src のビルド")
            src_success = run_command_with_output(
                "make -j$(nproc 2>/dev/null || echo 4)", 
                "srcのビルド",
                cwd=Path(KALDI_ROOT) / "src"
            )
            
            if src_success:
                print("\n🎉 Kaldiのビルドが完了しました！")
                
                # ビルド後の確認
                print_separator("ビルド結果確認")
                for binary_path in important_binaries:
                    full_path = Path(KALDI_ROOT) / binary_path
                    if full_path.exists():
                        print(f"✅ {binary_path}")
                    else:
                        print(f"❌ {binary_path}")
                        
            else:
                print("❌ srcのビルドに失敗しました。")
                print("手動で以下を実行してください：")
                print(f"cd {KALDI_ROOT}/src && make")
        else:
            print("ビルドをスキップしました。")
            print("\n手動でビルドする場合は以下を実行してください：")
            print(f"cd {KALDI_ROOT}/tools && make")
            print(f"cd {KALDI_ROOT}/src && make")
    
    else:
        print_separator("次のステップ")
        print("Kaldiは概ねビルド済みです。")
        print("アプリケーションを再テストしてみてください。")
        print("それでも問題が発生する場合は、以下を確認してください：")
        print("1. path.shが正しくソースされているか")
        print("2. PATHに正しいディレクトリが含まれているか")
        print("3. 必要な音響モデルファイルが存在するか")

    print_separator("完了")

if __name__ == "__main__":
    main()