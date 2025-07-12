#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleSpeech インストール確認とセットアップスクリプト
"""

import sys
import subprocess
import importlib.util
import os

def check_package_installed(package_name):
    """パッケージがインストールされているかチェック"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def run_command(command, description=""):
    """コマンドを実行し、結果を表示"""
    print(f"\n{'='*60}")
    print(f"実行中: {description}")
    print(f"コマンド: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"戻り値: {result.returncode}")
        if result.stdout:
            print(f"標準出力:\n{result.stdout}")
        if result.stderr:
            print(f"標準エラー:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"エラー: {e}")
        return False

def check_paddlespeech_imports():
    """PaddleSpeechの各モジュールのインポートをテスト"""
    modules_to_test = [
        ("paddlespeech.cli.asr.infer", "ASRExecutor"),
        ("paddlespeech.cli.text.infer", "TextExecutor"),
        ("paddlespeech.cli.align.infer", "AlignExecutor"),
    ]
    
    results = {}
    
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            class_obj = getattr(module, class_name)
            results[f"{module_name}.{class_name}"] = True
            print(f"✅ {module_name}.{class_name} - インポート成功")
        except ImportError as e:
            results[f"{module_name}.{class_name}"] = False
            print(f"❌ {module_name}.{class_name} - インポート失敗: {e}")
        except AttributeError as e:
            results[f"{module_name}.{class_name}"] = False
            print(f"❌ {module_name}.{class_name} - クラス取得失敗: {e}")
        except Exception as e:
            results[f"{module_name}.{class_name}"] = False
            print(f"❌ {module_name}.{class_name} - その他のエラー: {e}")
    
    return results

def check_python_version():
    """Python バージョンをチェック"""
    print(f"Python バージョン: {sys.version}")
    version_info = sys.version_info
    
    if version_info.major == 3 and version_info.minor >= 7:
        print("✅ Python バージョンは適切です (3.7以上)")
        return True
    else:
        print("❌ Python 3.7以上が必要です")
        return False

def install_paddlespeech():
    """PaddleSpeechをインストール"""
    print("\n" + "="*60)
    print("PaddleSpeech インストール開始")
    print("="*60)
    
    # PaddlePaddleのインストール
    print("\n1. PaddlePaddle をインストール...")
    paddle_install_cmd = "pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple"
    if not run_command(paddle_install_cmd, "PaddlePaddle インストール"):
        print("PaddlePaddle のインストールに失敗しました")
        return False
    
    # PaddleSpeechのインストール
    print("\n2. PaddleSpeech をインストール...")
    paddlespeech_install_cmd = "pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple"
    if not run_command(paddlespeech_install_cmd, "PaddleSpeech インストール"):
        print("PaddleSpeech のインストールに失敗しました")
        return False
    
    # 依存関係のインストール
    print("\n3. 追加の依存関係をインストール...")
    additional_deps = [
        "librosa",
        "soundfile",
        "scipy",
        "numpy",
        "PyYAML",
        "tqdm",
        "colorlog",
        "matplotlib",
        "seaborn"
    ]
    
    for dep in additional_deps:
        cmd = f"pip install {dep}"
        run_command(cmd, f"{dep} インストール")
    
    return True

def main():
    """メイン実行関数"""
    print("PaddleSpeech インストール確認とセットアップ")
    print("="*60)
    
    # Python バージョンチェック
    if not check_python_version():
        print("Python のアップグレードが必要です")
        return
    
    # PaddlePaddle のチェック
    print("\n1. PaddlePaddle の確認...")
    paddle_installed = check_package_installed("paddle")
    print(f"PaddlePaddle インストール状態: {'✅ インストール済み' if paddle_installed else '❌ 未インストール'}")
    
    # PaddleSpeech のチェック
    print("\n2. PaddleSpeech の確認...")
    paddlespeech_installed = check_package_installed("paddlespeech")
    print(f"PaddleSpeech インストール状態: {'✅ インストール済み' if paddlespeech_installed else '❌ 未インストール'}")
    
    # インポートテスト
    print("\n3. PaddleSpeech モジュールのインポートテスト...")
    if paddlespeech_installed:
        import_results = check_paddlespeech_imports()
        all_imports_successful = all(import_results.values())
        
        if all_imports_successful:
            print("\n✅ すべてのモジュールが正常にインポートできました！")
            print("PaddleSpeech は正常に動作する準備ができています。")
            return
        else:
            print("\n❌ 一部のモジュールでインポートエラーが発生しました")
    else:
        print("PaddleSpeech がインストールされていません")
    
    # インストール実行
    print("\n4. インストール実行...")
    user_input = input("PaddleSpeech をインストールしますか？ (y/n): ")
    
    if user_input.lower() in ['y', 'yes', 'はい']:
        if install_paddlespeech():
            print("\n✅ インストール完了！")
            print("再度インポートテストを実行します...")
            
            # 再テスト
            import_results = check_paddlespeech_imports()
            all_imports_successful = all(import_results.values())
            
            if all_imports_successful:
                print("\n🎉 PaddleSpeech のセットアップが完了しました！")
                print("以下のモジュールが利用可能です:")
                print("- paddlespeech.cli.asr.infer.ASRExecutor")
                print("- paddlespeech.cli.text.infer.TextExecutor")
                print("- paddlespeech.cli.align.infer.AlignExecutor")
            else:
                print("\n⚠️  インストールは完了しましたが、一部のモジュールでまだ問題があります")
                print("手動で依存関係を確認してください")
        else:
            print("\n❌ インストールに失敗しました")
    else:
        print("インストールをキャンセルしました")

if __name__ == "__main__":
    main()

# 使用例とテスト用コード
def test_paddlespeech():
    """PaddleSpeech の基本的な使用例とテスト"""
    print("\n" + "="*60)
    print("PaddleSpeech テスト実行")
    print("="*60)
    
    try:
        # ASRExecutor のテスト
        print("\n1. ASRExecutor のテスト...")
        from paddlespeech.cli.asr.infer import ASRExecutor
        asr = ASRExecutor()
        print("✅ ASRExecutor の初期化成功")
        
        # TextExecutor のテスト
        print("\n2. TextExecutor のテスト...")
        from paddlespeech.cli.text.infer import TextExecutor
        text = TextExecutor()
        print("✅ TextExecutor の初期化成功")
        
        # AlignExecutor のテスト
        print("\n3. AlignExecutor のテスト...")
        from paddlespeech.cli.align.infer import AlignExecutor
        align = AlignExecutor()
        print("✅ AlignExecutor の初期化成功")
        
        print("\n🎉 すべてのExecutorが正常に動作しています！")
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")

# テスト実行用の関数を呼び出す場合のコメントアウト
# test_paddlespeech()