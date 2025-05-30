import 'dart:async';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

// 定数の定義
const int MAX_FILE_SIZE_MB = 50;
const Duration TIMEOUT_DURATION = Duration(seconds: 120);
const String KALDI_SERVICE_URL = String.fromEnvironment(
  'KALDI_SERVICE_URL',
  defaultValue: 'http://localhost:8080/analyze'
);

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

class AnalysisDetails {
  final double pronunciation;
  final double intonation; 
  final double rhythm;

  AnalysisDetails({
    required this.pronunciation,
    required this.intonation, 
    required this.rhythm,
  });

  factory AnalysisDetails.fromJson(Map<String, dynamic> json) {
    return AnalysisDetails(
      pronunciation: json['pronunciation']?.toDouble() ?? 0.0,
      intonation: json['intonation']?.toDouble() ?? 0.0,
      rhythm: json['rhythm']?.toDouble() ?? 0.0,
    );
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

  factory AnalysisResult.error(String message) {
    return AnalysisResult(
      success: false,
      errorMessage: message,
    );
  }

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      success: json['success'] ?? false,
      score: json['score']?.toDouble(),
      feedback: json['feedback'],
      details: json['details'] != null 
        ? AnalysisDetails.fromJson(json['details'])
        : null,
      errorMessage: json['error_message'],
    );
  }
}

class AnalyzeService {
  final int maxFileSizeMB;
  final Duration timeoutDuration;
  final String kaldiServiceUrl;
  final _cache = <String, AnalysisResult>{};

  AnalyzeService({
    this.maxFileSizeMB = MAX_FILE_SIZE_MB,
    this.timeoutDuration = TIMEOUT_DURATION,
    this.kaldiServiceUrl = KALDI_SERVICE_URL,
  });

  Future<AnalysisResult> analyzeAudio(String audioFilePath) async {
    try {
      // ファイルの検証
      final file = File(audioFilePath);
      if (!await file.exists()) {
        throw AudioValidationException('Audio file does not exist');
      }

      // ファイルサイズチェック
      final fileSize = await file.length();
      if (fileSize > maxFileSizeMB * 1024 * 1024) {
        throw AudioValidationException('File size exceeds ${maxFileSizeMB}MB limit');
      }

      // APIリクエスト
      final request = http.MultipartRequest('POST', Uri.parse(kaldiServiceUrl))
        ..files.add(await http.MultipartFile.fromPath('audio_file', audioFilePath));

      final response = await request.send().timeout(timeoutDuration);
      final responseData = await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw KaldiServiceException(
          'Server returned error',
          statusCode: response.statusCode,
          originalError: responseData
        );
      }

      final result = AnalysisResult.fromJson(jsonDecode(responseData));
      _cache[audioFilePath] = result;
      return result;

    } on TimeoutException {
      throw KaldiServiceException('Request timed out after ${timeoutDuration.inSeconds} seconds');
    } on AudioValidationException {
      rethrow;
    } on KaldiServiceException {
      rethrow;
    } catch (e) {
      throw AudioAnalysisException('Unexpected error during analysis', e);
    }
  }

  void dispose() {
    _cache.clear();
  }
}