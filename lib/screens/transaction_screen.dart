import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart';
import 'package:http_parser/http_parser.dart';
import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'dart:convert';
import 'dart:js' as js;

// TranscriptionScreen ウィジェット定義
class TranscriptionScreen extends StatefulWidget {
  final String? targetText;
  
  const TranscriptionScreen({Key? key, this.targetText}) : super(key: key);

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

// TranscriptionScreen の状態管理
class _TranscriptionScreenState extends State<TranscriptionScreen>
    with TickerProviderStateMixin {
  final _audioRecorder = AudioRecorder();
  final TextEditingController _textController = TextEditingController();
  final Dio _dio = Dio();
  
  String? _recordingPath;
  Uint8List? _webRecordingData;
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

  @override
  void initState() {
    super.initState();
    _setupAnimation();
    _initializeRecorder();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _targetText = widget.targetText ?? 
                 (ModalRoute.of(context)?.settings.arguments as String?);
  }

  @override
  void dispose() {
    _recordingTimer?.cancel();
    _recordingAnimationController.dispose();
    _audioRecorder.dispose();
    _textController.dispose();
    super.dispose();
  }

  // アニメーション設定
  void _setupAnimation() {
    _recordingAnimationController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    _recordingAnimation = Tween<double>(begin: 1.0, end: 1.3)
        .animate(CurvedAnimation(parent: _recordingAnimationController, curve: Curves.easeInOut));
    _recordingAnimationController.repeat(reverse: true);
  }

  // レコーダーの初期化
  Future<void> _initializeRecorder() async {
    try {
      final hasPermission = await _audioRecorder.hasPermission();
      if (!hasPermission) {
        _showError('録音の権限がありません。設定から許可してください。');
      }
    } catch (e) {
      _showError('録音の初期化に失敗しました: $e');
    }
  }

  // 録音開始
  Future<void> _startRecording() async {
    try {
      _recordingPath = kIsWeb 
          ? 'recording_${DateTime.now().millisecondsSinceEpoch}.wav'
          : '${(await getTemporaryDirectory()).path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';

      await _audioRecorder.start(
        const RecordConfig(encoder: AudioEncoder.wav, bitRate: 128000, sampleRate: 44100),
        path: _recordingPath!,
      );

      setState(() {
        _isRecording = true;
        _hasRecorded = false;
        _recordingDuration = Duration.zero;
        _recordedText = '';
        _webRecordingData = null;
      });

      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        setState(() => _recordingDuration = Duration(seconds: timer.tick));
      });

    } catch (e) {
      _showError('録音開始に失敗しました: $e');
    }
  }

  // 録音停止
  Future<void> _stopRecording() async {
    try {
      _recordingTimer?.cancel();
      
      final result = await _audioRecorder.stop();
      
      setState(() {
        _isRecording = false;
        _hasRecorded = true;
        _isAnalyzing = true;
      });

      if (kIsWeb) {
        _webRecordingData = await _processWebRecording(result);
      }

      await _analyzePronunciation();
    } catch (e) {
      _showError('録音停止に失敗しました: $e');
      _resetRecording();
    }
  }

  // Web録音データ処理
  Future<Uint8List?> _processWebRecording(dynamic result) async {
    try {
      if (result == null) return null;
      if (result is Uint8List) return result;
      if (result is List<int>) return Uint8List.fromList(result);
      
      if (result is String) {
        if (result.startsWith('blob:')) {
          try {
            if (kIsWeb) {
              final arrayBuffer = await _blobUrlToArrayBuffer(result);
              if (arrayBuffer != null) {
                return arrayBuffer;
              }
            }
            return null;
          } catch (e) {
            print('blob URL処理エラー: $e');
            return null;
          }
        }
        
        if (result.startsWith('data:')) {
          final parts = result.split(',');
          if (parts.length == 2) {
            try {
              return base64Decode(parts[1]);
            } catch (e) {
              print('Data URL Base64デコードエラー: $e');
              return null;
            }
          }
        }
        
        try {
          return base64Decode(result);
        } catch (e) {
          print('Base64デコードエラー: $e');
          return null;
        }
      }
      return null;
    } catch (e) {
      print('Web録音データ処理エラー (top-level): $e');
      return null;
    }
  }

  // blob URLをUint8Listに変換 (JavaScript interop)
  Future<Uint8List?> _blobUrlToArrayBuffer(String blobUrl) async {
    try {
      if (!kIsWeb) return null;
      
      final completer = Completer<Uint8List?>();
      
      js.context.callMethod('eval', ['''
        (async function() {
          try {
            const response = await fetch('$blobUrl');
            const blob = await response.blob();
            
            const reader = new FileReader();
            reader.onload = function(e) {
              const arrayBuffer = e.target.result;
              const uint8Array = new Uint8Array(arrayBuffer);
              const array = Array.from(uint8Array);
              window.dartBlobResult = JSON.stringify(array);
            };
            reader.onerror = function(e) {
              console.error('FileReader error:', e);
              window.dartBlobResult = 'error';
            };
            reader.readAsArrayBuffer(blob);
          } catch (error) {
            console.error('Blob URL処理エラー:', error);
            window.dartBlobResult = 'error';
          }
        })();
      ''']);
      
      Timer(const Duration(seconds: 10), () {
        if (!completer.isCompleted) {
          completer.complete(null);
        }
      });
      
      Timer.periodic(const Duration(milliseconds: 100), (timer) {
        final result = js.context['dartBlobResult'];
        if (result != null) {
          timer.cancel();
          if (!completer.isCompleted) {
            if (result is String && result != 'null' && result != 'error') {
              try {
                final List<dynamic> arrayData = jsonDecode(result);
                final Uint8List uint8Data = Uint8List.fromList(arrayData.cast<int>());
                completer.complete(uint8Data);
              } catch (e) {
                print('JSON parsing error: $e');
                completer.complete(null);
              }
            } else {
              completer.complete(null);
            }
          }
          js.context['dartBlobResult'] = null;
        }
      });
      
      return await completer.future;
    } catch (e) {
      print('blob URL変換エラー: $e');
      return null;
    }
  }

  // API通信
  Future<void> _analyzePronunciation() async {
    try {
      FormData formData;
      
      if (kIsWeb) {
        if (_webRecordingData == null) throw Exception('Web録音データが無効です');
        
        formData = FormData.fromMap({
          'audio': MultipartFile.fromBytes( // フィールド名を 'audio' に修正
            _webRecordingData!,
            filename: 'audio.wav',
            contentType: MediaType('audio', 'wav'),
          ),
          'text': _targetText ?? '', // Web環境では文字列を直接渡す
          'model': 'default', // model フィールドも文字列として追加
        });
      } 
      else { // Web以外の環境（モバイルなど）の場合
        if (_recordingPath == null || !File(_recordingPath!).existsSync()) {
          throw Exception('録音ファイルが見つかりません');
        }
        formData = FormData.fromMap({
          'audio': await MultipartFile.fromFile(_recordingPath!, filename: 'audio.wav'), // フィールド名を 'audio' に修正
          'text': _targetText ?? '', // 非Web環境ではStringを直接渡す
          'model': 'default', // model フィールドも追加
        });
      }

      final response = await _dio.post(
        'http://localhost:8000/api/v1/analyze/', 
        data: formData,
        options: Options(
          headers: {'Accept': 'application/json'},
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 60),
        ),
      );

      if (response.statusCode == 200) {
        final data = response.data as Map<String, dynamic>;
        setState(() {
          _isAnalyzing = false;
          _recordedText = data['recognized_text'] ?? '音声が認識できませんでした';
        });
      } else {
        throw Exception('API Error: ${response.statusCode} - ${response.data}');
      }
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
        _recordedText = '認識に失敗しました';
      });
      _showError('音声認識に失敗しました: $e');
    }
  }

  // 録音状態のリセット
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

  // エラーメッセージ表示
  void _showError(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message), backgroundColor: Colors.red),
      );
    }
  }

  // 時間フォーマット変換
  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, "0");
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, "0");
    return "$minutes:$seconds";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF8674A1),
      appBar: AppBar(
        backgroundColor: const Color(0xFF8674A1),
        elevation: 0,
        title: const Text('発音練習', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back, color: Colors.white),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              _buildTargetTextCard(),
              const SizedBox(height: 32),
              _buildRecordingCard(),
              const SizedBox(height: 24),
              if (_hasRecorded && !_isAnalyzing) _buildResultCard(),
            ],
          ),
        ),
      ),
    );
  }

  // 練習テキストカードUI構築
  Widget _buildTargetTextCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Flexible(
                  child: Text('練習テキスト', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                ),
                IconButton(
                  icon: Icon(_isEditing ? Icons.check : Icons.edit),
                  onPressed: () {
                    setState(() {
                      if (_isEditing) {
                        _targetText = _textController.text;
                        _isEditing = false;
                      } else {
                        _isEditing = true;
                        _textController.text = _targetText ?? '';
                      }
                    });
                  },
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: BorderRadius.circular(12),
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
                    )
                  : Text(_targetText ?? '練習テキストが設定されていません'),
            ),
          ],
        ),
      ),
    );
  }

  // 録音カードUI構築
  Widget _buildRecordingCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            if (_isRecording)
              AnimatedBuilder(
                animation: _recordingAnimation,
                builder: (context, child) => Transform.scale(
                  scale: _recordingAnimation.value,
                  child: const Icon(Icons.mic, color: Colors.red, size: 60),
                ),
              )
            else if (_isAnalyzing)
              const CircularProgressIndicator()
            else
              const Icon(Icons.mic, color: Color(0xFF8674A1), size: 60),
            
            const SizedBox(height: 16),
            
            Text(
              _isRecording ? '録音中... ${_formatDuration(_recordingDuration)}' :
              _isAnalyzing ? '解析中...' :
              _hasRecorded ? '録音完了' : 'タップして録音開始',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            
            const SizedBox(height: 24),
            
            if (!_isAnalyzing)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_hasRecorded) ...[
                    Flexible(
                      child: OutlinedButton(
                        onPressed: _resetRecording,
                        child: const Text('もう一度'),
                      ),
                    ),
                    const SizedBox(width: 16),
                  ],
                  Flexible(
                    child: ElevatedButton(
                      onPressed: _isRecording ? _stopRecording : _startRecording,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isRecording ? Colors.red : Colors.purple,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                      ),
                      child: Text(_isRecording ? '録音停止' : '録音開始'),
                    ),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  // 結果比較カードUI構築
  Widget _buildResultCard() {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('結果比較', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            _buildComparisonText('目標テキスト', _targetText ?? '', Colors.blue),
            const SizedBox(height: 12),
            _buildComparisonText('認識結果', _recordedText, Colors.green),
          ],
        ),
      ),
    );
  }

  // 比較テキスト表示UI構築
  Widget _buildComparisonText(String label, String text, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: Colors.grey[600], fontSize: 12)),
        const SizedBox(height: 4),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: color.withOpacity(0.3)),
          ),
          child: Text(text, style: TextStyle(color: color.withOpacity(0.8))),
        ),
      ],
    );
  }
}
