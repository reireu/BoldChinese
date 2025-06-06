import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart';
import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'dart:convert';
import 'package:bold_chinese/utils/storage_helper.dart';

class TranscriptionScreen extends StatefulWidget {
  final String? targetText;
  
  const TranscriptionScreen({Key? key, this.targetText}) : super(key: key);

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen>
    with TickerProviderStateMixin {
  final _audioRecorder = AudioRecorder();
  final TextEditingController _textController = TextEditingController();
  final Dio _dio = Dio();
  
  String? _recordingPath;
  bool _isRecording = false;
  bool _isAnalyzing = false;
  bool _hasRecorded = false;
  bool _isEditing = false;
  String _recordedText = '';
  Duration _recordingDuration = Duration.zero;
  Timer? _recordingTimer;
  late AnimationController _recordingAnimationController;
  late Animation<double> _recordingAnimation;
  String? _targetText;
  
  // Web環境での録音データを保存
  Uint8List? _webRecordingData;

  @override
  void initState() {
    super.initState();
    _setupAnimation();
    _initializeRecorder();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;
    if (widget.targetText != null) {
      _targetText = widget.targetText;
    } else if (args is String) {
      _targetText = args;
    }
  }

  @override
  void dispose() {
    _recordingTimer?.cancel();
    _recordingAnimationController.dispose();
    _audioRecorder.dispose();
    _textController.dispose();
    super.dispose();
  }

  void _setupAnimation() {
    _recordingAnimationController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    _recordingAnimation = Tween<double>(
      begin: 1.0,
      end: 1.3,
    ).animate(CurvedAnimation(
      parent: _recordingAnimationController,
      curve: Curves.easeInOut,
    ));
    _recordingAnimationController.repeat(reverse: true);
  }

  Future<void> _initializeRecorder() async {
    try {
      final hasPermission = await _audioRecorder.hasPermission();
      if (!hasPermission) {
        print('録音の権限がありません。設定から許可してください。');
      }
    } catch (e) {
      print('エラー内容: $e');
    }
  }

  void _onSubmitText() {
    if (_textController.text.isNotEmpty) {
      setState(() {
        _targetText = _textController.text;
        _isEditing = false;
      });
    }
  }

  Future<void> _startRecording() async {
    try {
      String recordingPath;
      if (kIsWeb) {
        recordingPath = 'recording_${DateTime.now().millisecondsSinceEpoch}.wav';
      } else {
        final dir = await getTemporaryDirectory();
        recordingPath = '${dir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';
      }

      _recordingPath = recordingPath;
      _webRecordingData = null; // リセット

      await _audioRecorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          bitRate: 128000,
          sampleRate: 44100,
        ),
        path: recordingPath,
      );

      setState(() {
        _isRecording = true;
        _hasRecorded = false;
        _recordingDuration = Duration.zero;
        _recordedText = '';
      });

      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        setState(() {
          _recordingDuration = Duration(seconds: timer.tick);
        });
      });

    } catch (e) {
      print('録音開始エラー: $e');
      _showErrorSnackBar('録音の開始に失敗しました');
    }
  }

  Future<void> _stopRecording() async {
    try {
      _recordingTimer?.cancel();
      
      if (kIsWeb) {
        // Web用の録音停止処理
        final result = await _audioRecorder.stop();
        
        setState(() {
          _isRecording = false;
          _hasRecorded = true;
          _isAnalyzing = true;
        });
        
        // Web環境での録音データ処理
        if (result != null) {
          _webRecordingData = await _processWebRecordingData(result);
          
          if (_webRecordingData == null || _webRecordingData!.isEmpty) {
            throw Exception('有効な録音データを取得できませんでした');
          }
          
          // データ検証
          if (!_validateRecordingData(_webRecordingData!)) {
            throw Exception('録音データの検証に失敗しました');
          }
          
          print('Web録音データ処理完了: ${_webRecordingData!.length} bytes');
        } else {
          throw Exception('録音データが取得できませんでした');
        }
        
        await _analyzePronunciation();
      } else {
        // ネイティブ用の録音停止処理
        await _audioRecorder.stop();
        
        if (_recordingPath != null && File(_recordingPath!).existsSync()) {
          setState(() {
            _isRecording = false;
            _hasRecorded = true;
            _isAnalyzing = true;
          });
          await _analyzePronunciation();
        } else {
          throw Exception('Recording file not found');
        }
      }
    } catch (e) {
      print('録音停止エラー: $e');
      setState(() {
        _isRecording = false;
        _hasRecorded = false;
        _isAnalyzing = false;
      });
      _showErrorSnackBar('録音の停止に失敗しました: ${e.toString()}');
    }
  }

  /// Web環境での録音データを適切な形式に変換する
  Future<Uint8List?> _processWebRecordingData(dynamic result) async {
    try {
      if (result is String) {
        // Base64文字列の場合
        if (_isBase64String(result)) {
          print('Base64データとして処理中...');
          try {
            return base64Decode(result);
          } catch (e) {
            print('Base64デコードエラー: $e');
            // Base64デコードに失敗した場合、データURLの可能性をチェック
            if (result.startsWith('data:')) {
              return _processDataURL(result);
            }
            throw Exception('Base64デコードに失敗しました');
          }
        } 
        // Data URLの場合 (data:audio/wav;base64,...)
        else if (result.startsWith('data:')) {
          print('Data URLとして処理中...');
          return _processDataURL(result);
        }
        // 通常の文字列の場合（レアケース）
        else {
          print('文字列をUTF-8バイトとして処理中...');
          return Uint8List.fromList(utf8.encode(result));
        }
      } 
      // 既にバイナリデータの場合
      else if (result is Uint8List) {
        print('Uint8Listとして処理中...');
        return result;
      } 
      else if (result is List<int>) {
        print('List<int>からUint8Listに変換中...');
        return Uint8List.fromList(result);
      }
      // Blobや他のWeb APIオブジェクトの場合
      else if (result is List) {
        print('Listとして処理中...');
        try {
          return Uint8List.fromList(result.cast<int>());
        } catch (e) {
          print('List変換エラー: $e');
          return null;
        }
      }
      else {
        print('未対応の録音データ形式: ${result.runtimeType}');
        // 最後の手段として文字列変換を試行
        try {
          final stringData = result.toString();
          if (stringData.isNotEmpty && stringData != 'null') {
            return Uint8List.fromList(utf8.encode(stringData));
          }
        } catch (e) {
          print('フォールバック変換エラー: $e');
        }
        return null;
      }
    } catch (e) {
      print('録音データ処理エラー: $e');
      return null;
    }
  }

  /// Base64文字列かどうかを判定する
  bool _isBase64String(String str) {
    // 基本的なBase64文字列の判定
    if (str.isEmpty) return false;
    
    // Base64文字列は4の倍数の長さを持つ
    if (str.length % 4 != 0) return false;
    
    // Base64文字で構成されているかチェック
    final base64Pattern = RegExp(r'^[A-Za-z0-9+/]*={0,2}$');
    return base64Pattern.hasMatch(str);
  }

  /// Data URLを処理してバイナリデータを抽出する
  Uint8List? _processDataURL(String dataUrl) {
    try {
      // Data URLの形式: data:[<mime type>][;base64],<data>
      final parts = dataUrl.split(',');
      if (parts.length != 2) {
        throw Exception('無効なData URL形式です');
      }
      
      final header = parts[0];
      final data = parts[1];
      
      if (header.contains('base64')) {
        return base64Decode(data);
      } else {
        // Base64でない場合はURL デコードを試行
        final decodedData = Uri.decodeComponent(data);
        return Uint8List.fromList(utf8.encode(decodedData));
      }
    } catch (e) {
      print('Data URL処理エラー: $e');
      return null;
    }
  }

  /// 録音データの検証を行う
  bool _validateRecordingData(Uint8List data) {
    // 最小限のデータサイズチェック
    if (data.length < 44) {
      print('録音データが小さすぎます: ${data.length} bytes');
      return false;
    }
    
    // WAVファイルのヘッダーチェック（オプション）
    if (data.length >= 4) {
      final header = String.fromCharCodes(data.sublist(0, 4));
      if (header == 'RIFF') {
        print('WAVファイルヘッダーを検出しました');
        return true;
      }
    }
    
    // WebMやその他の形式の場合も有効とする
    print('録音データを有効として処理します: ${data.length} bytes');
    return true;
  }

  Future<void> _analyzePronunciation() async {
    try {
      if (!kIsWeb && (_recordingPath == null || !File(_recordingPath!).existsSync())) {
        setState(() {
          _isAnalyzing = false;
          _recordedText = '録音ファイルが見つかりません';
        });
        return;
      }

      FormData formData;
      
      if (kIsWeb) {
        // Web環境での処理
        if (_webRecordingData == null || _webRecordingData!.isEmpty) {
          throw Exception('Web環境での録音データが無効です');
        }
        
        // APIが期待する 'file' フィールド名を使用
        formData = FormData.fromMap({
          'file': MultipartFile.fromBytes(
            _webRecordingData!,
            filename: 'audio.wav',
            contentType: DioMediaType('audio', 'wav'),
          ),
          'text': _targetText ?? '',
        });
        
        print('Web音声データサイズ: ${_webRecordingData!.length} bytes');
      } else {
        // ネイティブ用: ファイルパスから読み込み
        formData = FormData.fromMap({
          'file': await MultipartFile.fromFile(
            _recordingPath!,
            filename: 'audio.wav',
            contentType: DioMediaType('audio', 'wav'),
          ),
          'text': _targetText ?? '',
        });
      }

      print('API送信開始: ${kIsWeb ? "Web環境" : "ネイティブ環境"}');

      final response = await _dio.post(
        'http://localhost:8000/api/v1/analyze-audio',
        data: formData,
        options: Options(
          headers: {
            'Accept': 'application/json',
          },
          // Content-Typeは自動設定されるため削除
          validateStatus: (status) => status! < 500,
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

      print('API応答ステータス: ${response.statusCode}');
      print('API応答データ: ${response.data}');

      if (response.statusCode == 200) {
        final responseData = response.data;
        if (responseData is Map<String, dynamic>) {
          final recognizedText = responseData['recognized_text'] as String? ?? '';
          setState(() {
            _isAnalyzing = false;
            _recordedText = recognizedText.isNotEmpty ? recognizedText : '音声が認識できませんでした';
          });
          print('音声認識結果: $recognizedText');
        } else {
          throw Exception('予期しないレスポンス形式: ${responseData.runtimeType}');
        }
      } else {
        final errorMessage = response.data is Map ? response.data.toString() : response.data;
        throw Exception('API error: ${response.statusCode} - $errorMessage');
      }
    } catch (e) {
      print('API通信エラー: $e');
      setState(() {
        _isAnalyzing = false;
        _recordedText = '認識に失敗しました';
      });
      
      String errorMessage = '音声認識に失敗しました';
      if (e is DioException) {
        switch (e.type) {
          case DioExceptionType.connectionTimeout:
          case DioExceptionType.sendTimeout:
          case DioExceptionType.receiveTimeout:
            errorMessage = '通信がタイムアウトしました';
            break;
          case DioExceptionType.connectionError:
            errorMessage = 'サーバーに接続できませんでした';
            break;
          case DioExceptionType.badResponse:
            final statusCode = e.response?.statusCode;
            final responseData = e.response?.data;
            if (statusCode == 422) {
              errorMessage = 'リクエストデータが不正です';
              print('422エラー詳細: $responseData');
            } else {
              errorMessage = 'サーバーエラーが発生しました ($statusCode)';
            }
            break;
          default:
            errorMessage = '通信エラー: ${e.message}';
        }
      } else if (e.toString().contains('API error')) {
        errorMessage = e.toString().replaceFirst('Exception: ', '');
      }
      _showErrorSnackBar(errorMessage);
    }
  }

  void _resetRecording() {
    setState(() {
      _isRecording = false;
      _hasRecorded = false;
      _isAnalyzing = false;
      _recordedText = '';
      _recordingDuration = Duration.zero;
      _recordingPath = null;
      _webRecordingData = null;
    });
    _recordingTimer?.cancel();
  }

  void _retryRecording() {
    _resetRecording();
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String minutes = twoDigits(duration.inMinutes.remainder(60));
    String seconds = twoDigits(duration.inSeconds.remainder(60));
    return "$minutes:$seconds";
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
          behavior: SnackBarBehavior.floating,
          duration: const Duration(seconds: 4),
          action: SnackBarAction(
            label: '閉じる',
            textColor: Colors.white,
            onPressed: () {
              ScaffoldMessenger.of(context).hideCurrentSnackBar();
            },
          ),
        ),
      );
    }
  }

  Widget _buildTargetTextCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.text_snippet_outlined,
                      color: Theme.of(context).primaryColor,
                      size: 24,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '練習テキスト',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: const Color(0xFF333333),
                      ),
                    ),
                  ],
                ),
                IconButton(
                  icon: Icon(_isEditing ? Icons.check : Icons.edit),
                  onPressed: () {
                    if (_isEditing) {
                      _onSubmitText();
                    } else {
                      setState(() {
                        _isEditing = true;
                        _textController.text = _targetText ?? '';
                      });
                    }
                  },
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey[200]!),
              ),
              child: _isEditing
                  ? TextField(
                      controller: _textController,
                      maxLines: null,
                      decoration: const InputDecoration(
                        border: InputBorder.none,
                        hintText: '練習したい文章を入力してください',
                      ),
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: const Color(0xFF333333),
                        height: 1.6,
                        fontSize: 16,
                      ),
                    )
                  : Text(
                      _targetText ?? '練習テキストが設定されていません',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: const Color(0xFF333333),
                        height: 1.6,
                        fontSize: 16,
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecordingStatus() {
    if (_isRecording) {
      return Column(
        children: [
          AnimatedBuilder(
            animation: _recordingAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: _recordingAnimation.value,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: Colors.red.withOpacity(0.2),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.mic,
                    color: Colors.red,
                    size: 40,
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: 16),
          Text(
            '録音中...',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: Colors.red,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            _formatDuration(_recordingDuration),
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              color: const Color(0xFF333333),
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      );
    } else if (_isAnalyzing) {
      return Column(
        children: [
          CircularProgressIndicator(
            valueColor: AlwaysStoppedAnimation<Color>(
              Theme.of(context).primaryColor,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            '解析中...',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: Theme.of(context).primaryColor,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      );
    } else {
      return Column(
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: Theme.of(context).primaryColor.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.mic,
              color: Theme.of(context).primaryColor,
              size: 40,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            _hasRecorded ? '録音完了' : 'タップして録音開始',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: const Color(0xFF333333),
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      );
    }
  }

  Widget _buildRecordingButton() {
    final bool isEnabled = !_isAnalyzing;
    final String buttonText = _isRecording ? '録音停止' : '録音開始';
    final Color buttonColor = _isRecording ? Colors.red : Theme.of(context).primaryColor;

    return ElevatedButton(
      onPressed: isEnabled ? (_isRecording ? _stopRecording : _startRecording) : null,
      style: ElevatedButton.styleFrom(
        backgroundColor: buttonColor,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        elevation: 2,
      ),
      child: Text(
        buttonText,
        style: const TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildRetryButton() {
    return OutlinedButton(
      onPressed: _retryRecording,
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        side: BorderSide(color: Colors.grey[400]!),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      child: Text(
        'もう一度',
        style: TextStyle(
          color: Colors.grey[700],
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _buildRecordingCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            _buildRecordingStatus(),
            const SizedBox(height: 32),
            if (!_isAnalyzing)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_hasRecorded) ...[
                    _buildRetryButton(),
                    const SizedBox(width: 16),
                  ],
                  _buildRecordingButton(),
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextComparison() {
    if (!_hasRecorded || _isAnalyzing) {
      return const SizedBox.shrink();
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '結果比較',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: const Color(0xFF333333),
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            // 目標テキスト
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '目標テキスト',
                  style: Theme.of(context).textTheme.labelMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.blue[50],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.blue[200]!),
                  ),
                  child: Text(
                    _targetText ?? '',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.blue[800],
                      height: 1.5,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            // 認識結果
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '認識結果',
                  style: Theme.of(context).textTheme.labelMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.green[50],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.green[200]!),
                  ),
                  child: Text(
                    _recordedText.isNotEmpty ? _recordedText : '認識中...',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.green[800],
                      height: 1.5,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHintCard() {
    if (_hasRecorded || _isRecording) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).primaryColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: Theme.of(context).primaryColor.withOpacity(0.2),
          width: 1,
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.info_outline,
            color: Theme.of(context).primaryColor,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              '上記のテキストを読み上げて、録音ボタンをタップしてください',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).primaryColor,
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF8674A1),
      appBar: AppBar(
        backgroundColor: const Color(0xFF8674A1),
        elevation: 0,
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(
            Icons.arrow_back,
            color: Colors.white,
            size: 28,
          ),
        ),
        title: Text(
          '発音練習',
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildTargetTextCard(),
                const SizedBox(height: 32),
                _buildRecordingCard(),
                _buildTextComparison(),
                const SizedBox(height: 24),
                _buildHintCard(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}