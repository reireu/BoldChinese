import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'dart:convert'; // JSONデコードのために追加

/// オーディオ分析サービス全般で発生する可能性のある例外の基底クラス
class AudioAnalysisException implements Exception {
  final String message;
  final dynamic originalError;
  AudioAnalysisException(this.message, [this.originalError]);

  @override
  String toString() {
    if (originalError != null) {
      return 'AudioAnalysisException: $message ($originalError)';
    }
    return 'AudioAnalysisException: $message';
  }
}

/// オーディオファイルのバリデーションエラーを示す例外
class AudioValidationException extends AudioAnalysisException {
  AudioValidationException(String message) : super(message);
}

/// Kaldiサービスとの通信中に発生したエラーを示す例外
class KaldiServiceException extends AudioAnalysisException {
  final int? statusCode; 

  KaldiServiceException(String message, {this.statusCode, dynamic originalError})
      : super(message, originalError);

  @override
  String toString() {
    String statusInfo = statusCode != null ? ' (Status: $statusCode)' : '';
    return 'KaldiServiceException: $message$statusInfo';
  }
}

class AnalysisResult {
  final bool success;
  final double? score;
  final String? feedback;
  final AnalysisDetails? details;
  final String? errorMessage;

  AnalysisResult({
    required this.success,
    this.score,
    this.feedback,
    this.details,
    this.errorMessage,
  });

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      success: json['success'] as bool,
      score: (json['score'] as num?)?.toDouble(), // null許容に対応
      feedback: json['feedback'] as String?,
      details: json['details'] != null
          ? AnalysisDetails.fromJson(json['details'])
          : null,
      errorMessage: json['error'] as String?,
    );
  }

  factory AnalysisResult.error(String message, {dynamic originalError}) {
    // エラー発生時に元のエラー情報を渡せるように改善
    return AnalysisResult(
      success: false,
      errorMessage: message + (originalError != null ? ' ($originalError)' : ''),
    );
  }

  @override
  String toString() {
    if (success) {
      return 'AnalysisResult(success: true, score: $score, feedback: "$feedback", details: $details)';
    } else {
      return 'AnalysisResult(success: false, errorMessage: "$errorMessage")';
    }
  }
}

class AnalysisDetails {
  final double pronunciation;
  final double intonation;
  final double rhythm;

  const AnalysisDetails({
    required this.pronunciation,
    required this.intonation,
    required this.rhythm,
  });

  factory AnalysisDetails.fromJson(Map<String, dynamic> json) {
    try {
      return AnalysisDetails(
        pronunciation: (json['pronunciation'] as num).toDouble(),
        intonation: (json['intonation'] as num).toDouble(),
        rhythm: (json['rhythm'] as num).toDouble(),
      );
    } catch (e) {
      throw FormatException('Invalid JSON format for AnalysisDetails: $e');
    }
  }

  Map<String, dynamic> toJson() => {
        'pronunciation': pronunciation,
        'intonation': intonation,
        'rhythm': rhythm,
      };

  bool isValid() {
    return pronunciation >= 0 &&
        pronunciation <= 100 &&
        intonation >= 0 &&
        intonation <= 100 &&
        rhythm >= 0 &&
        rhythm <= 100;
  }

  @override
  String toString() =>
      'AnalysisDetails(pronunciation: $pronunciation, intonation: $intonation, rhythm: $rhythm)';
}

class AnalyzeService {
  // 定数をより具体的に
  static const String kaldiServiceUrl = String.fromEnvironment(
    'KALDI_SERVICE_URL',
    defaultValue: 'http://localhost:8080/analyze'
  );
  
  // キャッシュの追加
  final _cache = <String, AnalysisResult>{};
  
  // リトライ機能の追加
 Future<AnalysisResult> analyzeAudioWithRetry(
    File audioFile, {
    int maxRetries = 3,
    Duration retryDelay = const Duration(seconds: 1),
  }) async {
    for (var i = 0; i < maxRetries; i++) {
      try {
        return await analyzeAudio(audioFile);
      } on KaldiServiceException catch (e) {
        if (i == maxRetries - 1) rethrow;
        await Future.delayed(retryDelay * (i + 1));
      }
    }
    throw KaldiServiceException('Max retries exceeded');
  }
  
    // タイムアウトの設定
    Future<void> preloadModels() async {
  }

  // メモリ解放
  void dispose() {
    _cache.clear();
  }
}

  /// メッセージをログに出力します。
  void _log(String message, {bool isError = false}) {
    if (isError) {
      // 実際には logger パッケージなどを利用
      print('ERROR: AnalyzeService - $message');
    } else {
      print('INFO: AnalyzeService - $message');
    }
  }

  /// オーディオファイルを分析し、AnalysisResultを返します。
  Future<AnalysisResult> analyzeAudio(File audioFile) async {
    _log('Starting audio analysis for: ${audioFile.path}');
    try {
      // ファイルバリデーション
      await _validateAudioFile(audioFile);

      // Kaldiサービスにリクエスト送信
      final Map<String, dynamic> rawResponse =
          await _sendToKaldiService(audioFile);

      // 結果の解析と加工
      return _processAnalysisResult(rawResponse);
    } on AudioValidationException catch (e) {
      _log('Audio validation error: ${e.message}', isError: true);
      return AnalysisResult.error('Invalid audio file: ${e.message}', originalError: e);
    } on KaldiServiceException catch (e) {
      _log('Kaldi service communication error: ${e.message}', isError: true);
      return AnalysisResult.error('Failed to communicate with analysis service: ${e.message}', originalError: e);
    } on TimeoutException catch (e) {
      _log('Request timed out: $e', isError: true);
      return AnalysisResult.error('Analysis request timed out. Please try again.', originalError: e);
    } catch (e) {
      _log('An unexpected error occurred: $e', isError: true);
      return AnalysisResult.error('An unexpected error occurred during analysis: $e', originalError: e);
    }
  }

  /// オーディオファイルのバリデーションを行います。
  /// 無効な場合は [AudioValidationException] をスローします。
  Future<void> _validateAudioFile(File file) async {
    final extension = path.extension(file.path).toLowerCase();
    _log('Validating file: ${file.path}, extension: $extension');

    if (!['.wav', '.mp3'].contains(extension)) {
      throw AudioValidationException('Unsupported audio file format: $extension. Only .wav and .mp3 are supported.');
    }

    // ファイルサイズのチェック
    final fileSize = await file.length(); 
    if (fileSize > maxFileSizeMB * 1024 * 1024) {
      throw AudioValidationException(
          'Audio file size exceeds the limit of ${maxFileSizeMB}MB.');
    }

    // MIMEタイプバリデーションのプレースホルダー
    // 実際のアプリケーションでは、file_pickerなどからMIMEタイプを取得して検証することが多い
    // String? mimeType = await file_picker_package.getMimeType(file.path);
    // if (mimeType != null && !['audio/wav', 'audio/mpeg'].contains(mimeType)) {
    //   throw AudioValidationException('Invalid MIME type for audio file.');
    // }

    _log('Audio file validation successful.');
  }

  /// Kaldiサービスにオーディオファイルを送信し、その生のレスポンスを返します。
  /// エラーが発生した場合は [KaldiServiceException] または [TimeoutException] をスローします。
  Future<Map<String, dynamic>> _sendToKaldiService(File audioFile) async {
    _log('Sending audio file to Kaldi service: ${audioFile.path}');
    try {
      var request = http.MultipartRequest('POST', Uri.parse(kaldiServiceUrl));
      request.files.add(
        await http.MultipartFile.fromPath('audio_file', audioFile.path),
      );

      final streamedResponse = await request.send().timeout(timeoutDuration);
      final response = await http.Response.fromStream(streamedResponse);

      _log('Received response from Kaldi service with status: ${response.statusCode}');

      if (response.statusCode != 200) {
        // エラーレスポンスボディをログに記録
        _log('Kaldi service returned error: ${response.body}', isError: true);
        throw KaldiServiceException(
            'Kaldi service returned an error. Status: ${response.statusCode}',
            statusCode: response.statusCode,
            originalError: response.body);
      }

      // レスポンスボディがJSON形式であることを期待
      try {
        return json.decode(response.body) as Map<String, dynamic>;
      } on FormatException catch (e) {
        _log('Failed to parse Kaldi service response as JSON: ${response.body}', isError: true);
        throw KaldiServiceException(
            'Failed to parse analysis result from service.',
            originalError: e);
      }
    } on SocketException catch (e) {
      throw KaldiServiceException('Network error communicating with Kaldi service: $e', originalError: e);
    } on TimeoutException catch (e) {
      throw TimeoutException('Request to Kaldi service timed out after ${timeoutDuration.inSeconds} seconds.', e.duration);
    } catch (e) {
      throw KaldiServiceException('Failed to communicate with Kaldi service: $e', originalError: e);
    }
  }

  /// Kaldiサービスからの生のレスポンスを解析し、[AnalysisResult] オブジェクトに変換します。
  AnalysisResult _processAnalysisResult(Map<String, dynamic> rawResponse) {
    _log('Processing raw analysis result.');
    try {
      // Kaldiからの生のレスポンス構造に応じて、AnalysisResult.fromJson に渡すデータを整形
      // 例: Kaldiが直接 'score', 'feedback', 'details' を返す場合
      // もしKaldiからのレスポンスが異なる構造の場合、ここで変換ロジックを追加する必要があります。
      // 例: rawResponse['result']['final_score'] のようなネストされた構造
      // 今回は、rawResponseが直接 AnalysisResult.fromJson に適合する形だと仮定します。
      final analysisResult = AnalysisResult.fromJson(rawResponse);

      // ここでさらにビジネスロジックに基づいてスコアやフィードバックを調整することも可能
      // 例: _calculateScore(rawResponse) などのメソッドは、JSONから取得した値を元に調整する場合に使用
      if (!analysisResult.success) {
        _log('Analysis result indicates failure: ${analysisResult.errorMessage}', isError: true);
        return AnalysisResult.error('Analysis was not successful: ${analysisResult.errorMessage}');
      }

      if (analysisResult.details != null && !analysisResult.details!.isValid()) {
         _log('Analysis details are out of valid range: ${analysisResult.details}', isError: true);
         // スコアや詳細の範囲外チェックが真であればエラーとして処理
         return AnalysisResult.error('Analysis details contain invalid values.');
      }

      _log('Analysis result processed successfully.');
      return analysisResult;
    } on FormatException catch (e) {
      _log('Failed to parse analysis result from Kaldi service: $e', isError: true);
      return AnalysisResult.error('Invalid analysis result format from service: $e', originalError: e);
    } catch (e) {
      _log('Error processing analysis result: $e', isError: true);
      return AnalysisResult.error('Error processing analysis result: $e', originalError: e);
    }
  }