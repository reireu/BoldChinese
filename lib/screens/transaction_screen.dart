import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart';
import 'dart:io';
import 'dart:async';

class TranscriptionScreen extends StatefulWidget {
  final String? targetText;
  const TranscriptionScreen({Key? key, this.targetText}) : super(key: key);

  @override
  State<TranscriptionScreen> createState() => _TranscriptionScreenState();
}

class _TranscriptionScreenState extends State<TranscriptionScreen>
    with TickerProviderStateMixin {
  // Use AudioRecorder instead of Record
  final _audioRecorder = AudioRecorder();
  String? _recordingPath;
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
    _setupAnimation();
    _initializeRecorder();
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
        // Handle permission request or show error
        print('Audio recording permission not granted');
      }
    } catch (e) {
      print('Error initializing recorder: $e');
    }
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
    super.dispose();
  }

  Future<void> _startRecording() async {
    try {
      final dir = await getTemporaryDirectory();
      _recordingPath = '${dir.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';

      // Use the new API with RecordConfig
      await _audioRecorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          bitRate: 128000,
          sampleRate: 44100,
        ),
        path: _recordingPath!,
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

      print('録音開始: $_targetText');
    } catch (e) {
      print('録音開始エラー: $e');
      _showErrorSnackBar('録音の開始に失敗しました');
    }
  }

  Future<void> _stopRecording() async {
    _recordingTimer?.cancel();
    try {
      await _audioRecorder.stop();
      setState(() {
        _isRecording = false;
        _hasRecorded = true;
        _isAnalyzing = true;
      });

      await _analyzePronunciation();
    } catch (e) {
      print('録音停止エラー: $e');
      setState(() {
        _isRecording = false;
        _hasRecorded = false;
        _isAnalyzing = false;
      });
      _showErrorSnackBar('録音の停止に失敗しました');
    }
  }

  Future<void> _analyzePronunciation() async {
    if (_recordingPath == null || !File(_recordingPath!).existsSync()) {
      setState(() {
        _isAnalyzing = false;
        _recordedText = '録音ファイルが見つかりません';
      });
      return;
    }

    try {
      final dio = Dio();
      final formData = FormData.fromMap({
        'audio': await MultipartFile.fromFile(_recordingPath!, filename: 'audio.wav'),
        'text': _targetText ?? '',
      });

      final response = await dio.post(
        'http://localhost:8000/api/analyze', 
        data: formData,
        options: Options(
          headers: {
            'Accept': 'application/json',
          },
          validateStatus: (status) => status! < 500,
        ),
      );

      if (response.statusCode == 200) {
        final recognizedText = response.data['recognized_text'] as String? ?? '';
        setState(() {
          _isAnalyzing = false;
          _recordedText = recognizedText;
        });
      } else {
        throw Exception('API error: ${response.statusCode}');
      }
    } catch (e) {
      print('API通信エラー: $e');
      setState(() {
        _isAnalyzing = false;
        _recordedText = '認識に失敗しました';
      });
      _showErrorSnackBar('音声認識に失敗しました');
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