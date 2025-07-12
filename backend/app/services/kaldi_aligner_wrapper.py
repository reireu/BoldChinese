import argparse
from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class KaldiConfig:
    kaldi_root: str = os.getenv("KALDI_ROOT", "/Users/serenakurashina/kaldi")
    
    # 修正: レシピの実行ルートは aidatatang_200zh/s5 に設定
    # ここに path.sh, steps, utils が存在する（またはシンボリックリンクされている）
    mandarin_recipe_root: str = os.getenv("MANDARIN_RECIPE_ROOT", "/Users/serenakurashina/kaldi/egs/aidatatang_200zh/s5") # ← ここを修正
    
    # モデルと言語データのルートディレクトリは aidatatang_asr を指す
    # この下に exp/chain/tdnn_1a_sp と data/lang_chain がある
    model_dir: str = os.getenv("MANDARIN_MODEL_PATH", "/Users/serenakurashina/kaldi/egs/mandarin_bn_bci/aidatatang_asr")
    
    # モデルのサブパスと言語のサブパスは model_dir からの相対パス
    acoustic_model: str = "aidatatang_asr/exp/chain/tdnn_1a_sp"
    lang_dir: str = "aidatatang_asr/data/lang_chain"

    def __post_init__(self):
        self.kaldi_root = Path(self.kaldi_root).resolve()
        self.mandarin_recipe_root = Path(self.mandarin_recipe_root).resolve()
        self.model_dir = Path(self.model_dir).resolve()

class KaldiAlignmentError(Exception):
    pass

class KaldiService:
    def __init__(self, config: KaldiConfig): 
        self.config = config 
        # Chainモデルを使用するのでnnet3のalignスクリプトを使用
        # このパスは mandarin_recipe_root からの相対パス
        self.align_script_path = "steps/online/nnet2/align.sh" 
        
    def check_kaldi_environment(self) -> Tuple[bool, List[str]]:
        logger.info("Kaldi環境（Mandarin対応）をチェック中...")
        errors = []
        
        try:
            required_paths = [
                (self.config.kaldi_root, "KALDI_ROOT"),
                (self.config.kaldi_root / "src" / "bin", "Kaldiバイナリディレクトリ"),
                
                # レシピの実行ルートのチェック
                (self.config.mandarin_recipe_root, "MANDARIN_RECIPE_ROOT"),
                (self.config.mandarin_recipe_root / "path.sh", "path.sh"),
                (self.config.mandarin_recipe_root / "utils", "utilsディレクトリ"), # utilsもレシピルート直下にあるべき
                
                # モデルデータのルートディレクトリのチェック
                (self.config.model_dir, "Mandarinモデルディレクトリ"),
            ]
            
            for path, description in required_paths:
                if not path.exists():
                    error_msg = f"{description}が見つかりません: {path}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                else:
                    logger.debug(f"{description}確認済み: {path}")
            
            # Kaldiバイナリのチェック (cwdはmandarin_recipe_root)
            binary_errors = self._check_kaldi_binaries()
            if binary_errors:
                errors.extend(binary_errors)
            
            # alignスクリプトのチェック (mandarin_recipe_root からの相対パス)
            align_script = self.config.mandarin_recipe_root / self.align_script_path
            if not align_script.exists():
                error_msg = f"アライメントスクリプトが見つかりません: {align_script}"
                logger.error(error_msg)
                errors.append(error_msg)
            
            # モデルファイルと言語ファイルのチェック (model_dir からの相対パス)
            model_errors = self._check_mandarin_model()
            if model_errors:
                errors.extend(model_errors)
            
            lang_errors = self._check_lang_directory()
            if lang_errors:
                errors.extend(lang_errors)
            
            if not errors:
                logger.info("Kaldi環境（Mandarin）チェック完了")
                return True, []
            else:
                logger.error(f"Kaldi環境チェックで{len(errors)}個のエラーが発生")
                return False, errors
                
        except Exception as e:
            error_msg = f"Kaldi環境チェック中の予期しないエラー: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return False, errors
    
    def _check_kaldi_binaries(self) -> List[str]:
        errors = []
        required_binaries = [
            "compute-mfcc-feats", 
            "nnet3-align-compiled",  
            "nnet3-latgen-faster",   
            "copy-feats", 
            "apply-cmvn",
            "ali-to-phones",
            "lattice-align-words"
        ]
        path_sh = self.config.mandarin_recipe_root / "path.sh"
        
        if not path_sh.exists():
            error_msg = f"path.shが見つかりません: {path_sh}"
            errors.append(error_msg)
            return errors
        
        for binary in required_binaries:
            try:
                # path.sh を source する際の cwd (カレントワーキングディレクトリ) を mandarin_recipe_root にする
                cmd = f"source {path_sh} && which {binary}"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10,
                    cwd=self.config.mandarin_recipe_root # ← ここが重要
                )
                if result.returncode != 0:
                    error_msg = f"Kaldiバイナリが見つかりません: {binary}"
                    if result.stderr:
                        error_msg += f" (stderr: {result.stderr.strip()})"
                    logger.error(error_msg)
                    errors.append(error_msg)
                else:
                    logger.debug(f"バイナリ確認済み: {binary} -> {result.stdout.strip()}")
            except subprocess.TimeoutExpired:
                error_msg = f"バイナリチェックタイムアウト: {binary}"
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"バイナリチェックエラー {binary}: {e}"
                errors.append(error_msg)
        
        return errors
    
    def _check_mandarin_model(self) -> List[str]:
        errors = []
        
        # model_dir から acoustic_model の相対パスを結合
        acoustic_model_path = self.config.model_dir / self.config.acoustic_model
        
        if not acoustic_model_path.exists():
            error_msg = f"音響モデルディレクトリが見つかりません: {acoustic_model_path}"
            errors.append(error_msg)
            return errors
        
        expected_model_files = [
            "final.mdl",
            "tree", 
        ]
        
        if "nnet3" in self.config.acoustic_model or "chain" in self.config.acoustic_model:
            # Chainモデルの場合に必要なファイル
            expected_model_files.extend([
                "final.raw",  
                "cmvn_opts",
                "graph/HCLG.fst", # HCLG.fstはgraphサブディレクトリにあることが多い
                "graph/words.txt" # words.txtもgraphサブディレクトリにあることが多い
            ])
        
        missing_files = []
        for file_name in expected_model_files:
            file_path = acoustic_model_path / file_name
            if not file_path.exists():
                missing_files.append(str(file_path))
            else:
                logger.debug(f"モデルファイル確認済み: {file_path}")
        
        if missing_files:
            error_msg = f"音響モデルファイルが不足: {missing_files}"
            errors.append(error_msg)
        else:
            logger.info(f"音響モデル確認完了: {acoustic_model_path}")
        
        return errors
    
    def _check_lang_directory(self) -> List[str]:
        errors = []
        
        # model_dir から lang_dir の相対パスを結合
        lang_path = self.config.model_dir / self.config.lang_dir
        
        if not lang_path.exists():
            error_msg = f"言語ディレクトリが見つかりません: {lang_path}"
            errors.append(error_msg)
            return errors
        
        required_lang_files = [
            "phones.txt",
            "words.txt", 
            "L.fst",
            "oov.int",
            "topo"
        ]
        
        missing_files = []
        for file_name in required_lang_files:
            file_path = lang_path / file_name
            if not file_path.exists():
                missing_files.append(str(file_path))
            else:
                logger.debug(f"言語ファイル確認済み: {file_path}")
        
        if missing_files:
            error_msg = f"言語ディレクトリファイルが不足: {missing_files}"
            errors.append(error_msg)
        else:
            logger.info(f"言語ディレクトリ確認完了: {lang_path}")
        
        return errors
    
    def validate_audio_file(self, audio_path: str) -> bool:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")
        
        if path.stat().st_size == 0:
            raise ValueError(f"音声ファイルが空です: {audio_path}")
            
        logger.info(f"音声ファイル検証完了: {audio_path} ({path.stat().st_size} bytes)")
        return True
    
    def convert_audio_format(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        try:
            sox_check = subprocess.run(["which", "sox"], capture_output=True, text=True)
            if sox_check.returncode != 0:
                return False, "SoXが見つかりません。brew install soxでインストールしてください。"
            
            cmd = ["sox", input_path, "-r", "16000", "-c", "1", "-b", "16", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_msg = f"音声変換エラー (code: {result.returncode}): {result.stderr}"
                logger.error(error_msg)
                return False, error_msg
                
            if result.stderr and "WARN" in result.stderr:
                logger.warning(f"音声変換警告: {result.stderr.strip()}")
                
            output_path_obj = Path(output_path)
            if not output_path_obj.exists():
                return False, f"変換後のファイルが作成されませんでした: {output_path}"
            
            if output_path_obj.stat().st_size == 0:
                return False, f"変換後のファイルが空です: {output_path}"
                
            logger.info(f"音声変換完了: {output_path} ({output_path_obj.stat().st_size} bytes)")
            return True, ""
            
        except subprocess.TimeoutExpired:
            return False, "音声変換タイムアウト（30秒）"
        except FileNotFoundError:
            return False, "SoXコマンドが見つかりません"
        except Exception as e:
            return False, f"音声変換エラー: {e}"
    
    def prepare_kaldi_files(self, work_dir: Path, audio_path: str, text: str) -> Dict[str, Path]:
        """Kaldiの標準的なディレクトリ構造を作成"""
        data_dir = work_dir / "data" / "test"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # text ファイル
        text_file = data_dir / "text"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(f"utt1 {text}\n")
        files["text"] = text_file
        
        # wav.scp ファイル
        wav_scp = data_dir / "wav.scp"
        with open(wav_scp, "w") as f:
            f.write(f"utt1 {audio_path}\n")
        files["wav_scp"] = wav_scp
        
        # utt2spk ファイル
        utt2spk = data_dir / "utt2spk"
        with open(utt2spk, "w") as f:
            f.write("utt1 spk1\n")
        files["utt2spk"] = utt2spk
        
        # spk2utt ファイル
        spk2utt = data_dir / "spk2utt"
        with open(spk2utt, "w") as f:
            f.write("spk1 utt1\n")
        files["spk2utt"] = spk2utt
        
        logger.info(f"Kaldiファイル準備完了: {data_dir}")
        return files
    
    def run_kaldi_alignment(self, work_dir: Path, lang_dir: str, model_dir: str) -> subprocess.CompletedProcess:
        """nnet3/chainモデル用のアライメント実行"""
        align_script = self.config.mandarin_recipe_root / self.align_script_path
        alignment_output = work_dir / "exp" / "chain_ali"
        alignment_output.mkdir(parents=True, exist_ok=True)
        
        path_sh = self.config.mandarin_recipe_root / "path.sh"
        data_test_dir = work_dir / "data" / "test"
        
        cmd = (
            f"source {path_sh} && "
            f"bash {align_script} "
            f"--nj 1 --cmd run.pl "
            f"--use-gpu false "
            f"{data_test_dir} {lang_dir} {model_dir} {alignment_output}"
        )
        
        logger.info(f"Mandarin音声アライメント実行中...")
        logger.info(f"コマンド: {cmd}")
        logger.info(f"作業ディレクトリ: {self.config.mandarin_recipe_root}")
        
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=self.config.mandarin_recipe_root, timeout=300
        )
    
    def parse_alignment_result(self, alignment_dir: Path, original_text: str) -> Dict[str, Any]:
        try:
            ali_files = list(alignment_dir.glob("ali.*.gz"))
            if not ali_files:
                raise KaldiAlignmentError(f"アライメントファイルが見つかりません: {alignment_dir}")
            
            ali_file = ali_files[0]
            ctm_file = alignment_dir / "ctm"
            model_path = Path(self.config.model_dir) / self.config.acoustic_model / "final.mdl"
            
            self._generate_ctm_file(ali_file, model_path, ctm_file)
            
            alignments = self._parse_ctm_file(ctm_file)
            
            return self._calculate_scores(alignments, original_text)
            
        except Exception as e:
            logger.error(f"アライメント結果解析エラー: {str(e)}")
            return self._create_fallback_result(f"結果解析エラー: {str(e)}")
    
    def _generate_ctm_file(self, ali_file: Path, model_path: Path, ctm_file: Path):
        path_sh = self.config.mandarin_recipe_root / "path.sh"
        
        cmd = (
            f"source {path_sh} && "
            f"gunzip -c {ali_file} | "
            f"ali-to-phones --ctm-output {model_path} ark:- > {ctm_file}"
        )
        
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
            cwd=self.config.mandarin_recipe_root
        )
        if result.returncode != 0:
            raise KaldiAlignmentError(f"CTM生成失敗: {result.stderr}")
    
    def _parse_ctm_file(self, ctm_file: Path) -> List[Dict[str, Any]]:
        alignments = []
        if ctm_file.exists():
            with open(ctm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # CTM format: <uttid> <file> <start> <duration> <phone> [<confidence>]
                        # Kaldi's ali-to-phones with --ctm-output can sometimes omit confidence for phones
                        # Ensure we handle both cases or assign a default.
                        # Assuming the phone is at index 4 if confidence is present, else index 3
                        phone_or_word_idx = 4 if len(parts) > 4 else 3
                        alignments.append({
                            "phone": parts[phone_or_word_idx],
                            "start": float(parts[2]),
                            "end": float(parts[2]) + float(parts[3]),
                            "confidence": float(parts[5]) if len(parts) > 5 else 1.0 # default to 1.0 if not present
                        })
        return alignments
    
    def _calculate_scores(self, alignments: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        if not alignments:
            return self._create_fallback_result("アライメントデータが空です")
        
        base_score = 75.0
        
        if len(alignments) > 0:
            duration_variation = np.std([a['end'] - a['start'] for a in alignments])
            if duration_variation < 0.1:
                base_score += 10
            elif duration_variation > 0.5:
                base_score -= 15
        
        # Consider confidence scores in overall calculation
        confidence_scores = [a['confidence'] for a in alignments]
        if confidence_scores:
            mean_confidence = np.mean(confidence_scores)
            base_score = base_score * (0.8 + 0.2 * mean_confidence) # Scale by confidence, with 0.8 as baseline
        
        pronunciation_score = min(100, max(0, base_score))
        intonation_score = min(100, max(0, base_score - 5))
        rhythm_score = min(100, max(0, base_score - 3))
        
        return {
            "success": True,
            "score": round(pronunciation_score, 2),
            "feedback": self._generate_feedback(pronunciation_score),
            "details": {
                "pronunciation": round(pronunciation_score, 2),
                "intonation": round(intonation_score, 2),
                "rhythm": round(rhythm_score, 2),
                "alignment_count": len(alignments)
            },
            "recognized_text": original_text,
            "model_type": "mandarin_chain"
        }
    
    def _generate_feedback(self, score: float) -> str:
        if score >= 95:
            return "完璧な発音です！ネイティブレベルの流暢さです。"
        elif score >= 85:
            return "素晴らしい発音です！中国語の音調も正確です。"
        elif score >= 75:
            return "良好な発音です。音調の練習でさらに向上できます。"
        elif score >= 65:
            return "基本的な発音は良好です。声調の区別を意識して練習しましょう。"
        elif score >= 50:
            return "発音の改善が必要です。基本的な音素と声調から練習しましょう。"
        else:
            return "発音の練習が必要です。ピンインの基礎から始めることをお勧めします。"
    
    def _create_fallback_result(self, error_message: str, detailed_errors: List[str] = None) -> Dict[str, Any]:
        logger.warning(f"フォールバック実行: {error_message}")
        
        result = {
            "success": False,
            "score": 0.0,
            "feedback": "音声分析に失敗しました。Mandarinモデルの設定を確認してください。",
            "details": {
                "pronunciation": 0.0, 
                "intonation": 0.0, 
                "rhythm": 0.0,
                "alignment_count": 0
            },
            "error_message": error_message,
            "recognized_text": "認識できませんでした",
            "model_type": "fallback"
        }
        
        if detailed_errors:
            result["detailed_errors"] = detailed_errors
            
        return result
    
    def run_analysis(self, audio_path: str, text: str) -> Dict[str, Any]:
        work_dir = None
        
        try:
            env_ok, env_errors = self.check_kaldi_environment()
            if not env_ok:
                logger.warning("Mandarin環境が利用できないため、フォールバックモードで動作")
                error_msg = f"Mandarin環境が正しく設定されていません。詳細: {'; '.join(env_errors[:3])}"
                return self._create_fallback_result(error_msg, env_errors)
            
            try:
                self.validate_audio_file(audio_path)
            except Exception as e:
                return self._create_fallback_result(f"音声ファイル検証失敗: {str(e)}")
            
            work_dir = Path(tempfile.mkdtemp(prefix="kaldi_mandarin_"))
            logger.info(f"作業ディレクトリ作成: {work_dir}")
            
            converted_audio = work_dir / "audio.wav"
            convert_ok, convert_error = self.convert_audio_format(audio_path, str(converted_audio))
            if not convert_ok:
                return self._create_fallback_result(f"音声ファイル変換失敗: {convert_error}")
            
            self.prepare_kaldi_files(work_dir, str(converted_audio), text)
            
            lang_dir = str(self.config.model_dir / self.config.lang_dir)
            model_dir = str(self.config.model_dir / self.config.acoustic_model)
            
            for dir_path, name in [(lang_dir, "言語ディレクトリ"), (model_dir, "音響モデルディレクトリ")]:
                if not Path(dir_path).exists():
                    return self._create_fallback_result(f"{name}が見つかりません: {dir_path}")
            
            result = self.run_kaldi_alignment(work_dir, lang_dir, model_dir)
            
            if result.returncode == 0:
                logger.info("Mandarin音声アライメント成功")
                return self.parse_alignment_result(work_dir / "exp" / "chain_ali", text)
            else:
                error_msg = f"アライメント失敗 (code: {result.returncode})"
                if result.stderr:
                    error_msg += f": {result.stderr[:500]}"
                if result.stdout:
                    error_msg += f" (stdout: {result.stdout[:200]})"
                logger.error(error_msg)
                return self._create_fallback_result(error_msg)
                
        except subprocess.TimeoutExpired:
            return self._create_fallback_result("処理タイムアウト（音声アライメントは時間がかかる場合があります）")
        except Exception as e:
            logger.error(f"予期しないエラー: {str(e)}", exc_info=True)
            return self._create_fallback_result(f"予期しないエラー: {str(e)}")
        finally:
            if work_dir and work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                    logger.info(f"作業ディレクトリ削除: {work_dir}")
                except Exception as e:
                    logger.error(f"クリーンアップエラー: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Kaldi音声アライメントツール（Mandarin対応）')
    parser.add_argument('--audio', required=True, help='音声ファイルパス')
    parser.add_argument('--text', required=True, help='テキスト')
    parser.add_argument('--kaldi-root', default=os.getenv('KALDI_ROOT', '/Users/serenakurashina/kaldi'),
                       help='Kaldiルートディレクトリ')
    parser.add_argument('--recipe-root', default=os.getenv('MANDARIN_RECIPE_ROOT', 
                       '/Users/serenakurashina/kaldi/egs/mandarin_bn_bci/aidatatang_asr'), 
                       help='Mandarinレシピルートディレクトリ')
    parser.add_argument('--model-dir', default=os.getenv('MANDARIN_MODEL_PATH', 
                       '/Users/serenakurashina/kaldi/egs/mandarin_bn_bci/aidatatang_asr'), 
                       help='Mandarinモデルディレクトリ')
    parser.add_argument('--acoustic-model', default='aidatatang_asr/exp/chain/tdnn_1a_sp', 
                       help='音響モデルパス')
    parser.add_argument('--lang-dir', default='aidatatang_asr/data/lang_chain', 
                       help='言語ディレクトリパス')
    args = parser.parse_args()
    
    logger.info(f"引数: audio={args.audio}, text={args.text}, "
                f"kaldi_root={args.kaldi_root}, recipe_root={args.recipe_root}, "
                f"model_dir={args.model_dir}, acoustic_model={args.acoustic_model}, "
                f"lang_dir={args.lang_dir}")
    
    config = KaldiConfig(
        kaldi_root=args.kaldi_root,
        mandarin_recipe_root=args.recipe_root,
        model_dir=args.model_dir,
        acoustic_model=args.acoustic_model,
        lang_dir=args.lang_dir
    )
    service = KaldiService(config)
    
    result = service.run_analysis(args.audio, args.text)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()