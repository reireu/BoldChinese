import 'package:flutter/material.dart';
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_widgets.dart';
import 'dart:async';

class TranscriptionScreen extends StatefulWidget {
  final String? targetText;
  const TranscriptionScreen({Key? key, this.targetText}) : super(key: key);

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen>
    with TickerProviderStateMixin {
  bool _isRecording = false;
  bool _isAnalyzing = false;
  bool _hasRecorded = false;
  String _recordedText = '';
  Duration _recordingDuration = Duration.zero;
  Timer? _recordingTimer;
  late AnimationController _recordingAnimationController;
  late Animation<double> _recordingAnimation;
  String? _targetText;

  @override
  void initState() {
    super.initState();
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

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // arguments経由でも受け取れるように
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
    super.dispose();
  }

  void _startRecording() {
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

    // TODO: 実際の録音開始処理をここに追加
    print('録音開始: $_targetText');
  }

  void _stopRecording() {
    _recordingTimer?.cancel();
    setState(() {
      _isRecording = false;
      _hasRecorded = true;
      _isAnalyzing = true;
    });

    // TODO: 録音停止と解析開始処理（API連携など）
    _simulateAnalysis();
  }

void _simulateAnalysis() async {
  // TODO: 録音ファイルのパスを取得する処理を追加
  final audioFilePath = '/path/to/recorded_audio.wav'; // 実際の録音ファイルパスに置き換えてください

  try {
    setState(() {
      _isAnalyzing = true;
    });

    // DioでAPIに音声ファイルを送信
    final dio = Dio();
    final formData = FormData.fromMap({
      'audio': await MultipartFile.fromFile(audioFilePath, filename: 'audio.wav'),
      'text': _targetText ?? '',
    });

    final response = await dio.post(
      'http://localhost:8000/api/recognize', // 実際のAPIエンドポイントに変更
      data: formData,
    );

    final recognizedText = response.data['recognized_text'] as String? ?? '';

    setState(() {
      _isAnalyzing = false;
      _recordedText = recognizedText;
    });
  } catch (e) {
    setState(() {
      _isAnalyzing = false;
      _recordedText = '認識に失敗しました';
    });
    print('認識エラー: $e');
  }
}

  void _resetRecording() {
    setState(() {
      _isRecording = false;
      _hasRecorded = false;
      _isAnalyzing = false;
      _recordedText = '';
      _recordingDuration = Duration.zero;
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

  Widget _buildTextComparison() {
    if (!_hasRecorded || _isAnalyzing) {
      return Container();
    }

    return Container(
      margin: const EdgeInsets.only(top: 24),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '結果比較',
            style: FlutterFlowTheme.of(context).titleMedium.override(
              fontFamily: 'Manrope',
              color: const Color(0xFF333333),
              fontWeight: FontWeight.bold,
              letterSpacing: 0.0,
            ),
          ),
          const SizedBox(height: 16),
          // 目標テキスト
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '目標テキスト',
                style: FlutterFlowTheme.of(context).labelMedium.override(
                  fontFamily: 'Manrope',
                  color: Colors.grey[600],
                  letterSpacing: 0.0,
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
                  style: FlutterFlowTheme.of(context).bodyMedium.override(
                    fontFamily: 'Manrope',
                    color: Colors.blue[800],
                    letterSpacing: 0.0,
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
                style: FlutterFlowTheme.of(context).labelMedium.override(
                  fontFamily: 'Manrope',
                  color: Colors.grey[600],
                  letterSpacing: 0.0,
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
                  style: FlutterFlowTheme.of(context).bodyMedium.override(
                    fontFamily: 'Manrope',
                    color: Colors.green[800],
                    letterSpacing: 0.0,
                    height: 1.5,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = FlutterFlowTheme.of(context);

    return Scaffold(
      backgroundColor: const Color(0xFF8674A1),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ヘッダー
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(
                        Icons.arrow_back,
                        color: Colors.white,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '発音練習',
                      style: theme.headlineSmall.override(
                        fontFamily: 'Manrope',
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.0,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 32),

                // 練習テキスト表示
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.text_snippet_outlined,
                              color: const Color(0xFF8674A1),
                              size: 24,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              '練習テキスト',
                              style: theme.titleMedium.override(
                                fontFamily: 'Manrope',
                                color: const Color(0xFF333333),
                                fontWeight: FontWeight.bold,
                                letterSpacing: 0.0,
                              ),
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
                          child: Text(
                            _targetText ?? '',
                            style: theme.bodyLarge.override(
                              fontFamily: 'Manrope',
                              color: const Color(0xFF333333),
                              letterSpacing: 0.0,
                              height: 1.6,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // 録音コントロール
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(32),
                    child: Column(
                      children: [
                        // ステータス表示
                        if (_isRecording)
                          Column(
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
                                style: theme.titleMedium.override(
                                  fontFamily: 'Manrope',
                                  color: Colors.red,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.0,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                _formatDuration(_recordingDuration),
                                style: theme.headlineMedium.override(
                                  fontFamily: 'Manrope',
                                  color: const Color(0xFF333333),
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.0,
                                ),
                              ),
                            ],
                          )
                        else if (_isAnalyzing)
                          Column(
                            children: [
                              const CircularProgressIndicator(
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  Color(0xFF8674A1),
                                ),
                              ),
                              const SizedBox(height: 16),
                              Text(
                                '解析中...',
                                style: theme.titleMedium.override(
                                  fontFamily: 'Manrope',
                                  color: const Color(0xFF8674A1),
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.0,
                                ),
                              ),
                            ],
                          )
                        else
                          Column(
                            children: [
                              Container(
                                width: 80,
                                height: 80,
                                decoration: BoxDecoration(
                                  color: const Color(0xFF8674A1).withOpacity(0.1),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.mic,
                                  color: Color(0xFF8674A1),
                                  size: 40,
                                ),
                              ),
                              const SizedBox(height: 16),
                              Text(
                                _hasRecorded ? '録音完了' : 'タップして録音開始',
                                style: theme.titleMedium.override(
                                  fontFamily: 'Manrope',
                                  color: const Color(0xFF333333),
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.0,
                                ),
                              ),
                            ],
                          ),

                        const SizedBox(height: 32),

                        // 録音ボタン
                        if (!_isAnalyzing)
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              if (_hasRecorded) ...[
                                // リトライボタン
                                FFButtonWidget(
                                  onPressed: _retryRecording,
                                  text: 'もう一度',
                                  options: FFButtonOptions(
                                    width: 120,
                                    height: 56,
                                    color: Colors.grey[100],
                                    textStyle: theme.titleMedium.override(
                                      fontFamily: 'Manrope',
                                      color: Colors.grey[700],
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 0.0,
                                    ),
                                    elevation: 0,
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                      width: 1,
                                    ),
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                ),
                                const SizedBox(width: 16),
                              ],
                              // メイン録音ボタン
                              FFButtonWidget(
                                onPressed: _isRecording ? _stopRecording : _startRecording,
                                text: _isRecording
                                    ? '録音停止'
                                    : (_hasRecorded ? '再録音' : '録音開始'),
                                options: FFButtonOptions(
                                  width: _hasRecorded ? 120 : 200,
                                  height: 56,
                                  color: _isRecording ? Colors.red : const Color(0xFF8674A1),
                                  textStyle: theme.titleMedium.override(
                                    fontFamily: 'Manrope',
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.0,
                                  ),
                                  elevation: 0,
                                  borderRadius: BorderRadius.circular(16),
                                ),
                              ),
                            ],
                          ),
                      ],
                    ),
                  ),
                ),

                // 結果比較表示
                _buildTextComparison(),

                const SizedBox(height: 24),

                // 使い方のヒント
                if (!_hasRecorded && !_isRecording)
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.2),
                        width: 1,
                      ),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.info_outline,
                          color: Colors.white.withOpacity(0.8),
                          size: 24,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            '上記のテキストを読み上げて、録音ボタンをタップしてください',
                            style: theme.bodySmall.override(
                              fontFamily: 'Manrope',
                              color: Colors.white.withOpacity(0.9),
                              letterSpacing: 0.0,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
