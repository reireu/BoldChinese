import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:dio/dio.dart';
import 'package:dio/io.dart';
import 'package:http/http.dart' as http;
import 'package:connectivity_plus/connectivity_plus.dart';
import 'dart:math' show max;

// === 定数定義（既存コードとの互換性のため） ===
const int MAX_FILE_SIZE_MB = 50;
const Duration TIMEOUT_DURATION = Duration(seconds: 120);
const Duration CONNECT_TIMEOUT_DURATION = Duration(seconds: 30);
const Duration RECEIVE_TIMEOUT_DURATION = Duration(seconds: 180);
const int MAX_RETRY_ATTEMPTS = 3;
const String KALDI_SERVICE_URL = String.fromEnvironment(
  'KALDI_SERVICE_URL',
  defaultValue: 'http://localhost:8000/analyze'
);

// === 設定クラス ===
class AnalysisConfig {
  final String serviceUrl;
  final int maxFileSizeMB;
  final Duration connectTimeout;
  final Duration receiveTimeout;
  final Duration sendTimeout;
  final int maxRetries;
  
  const AnalysisConfig({
    this.serviceUrl = KALDI_SERVICE_URL,
    this.maxFileSizeMB = MAX_FILE_SIZE_MB,
    this.connectTimeout = CONNECT_TIMEOUT_DURATION,
    this.receiveTimeout = RECEIVE_TIMEOUT_DURATION,
    this.sendTimeout = TIMEOUT_DURATION,
    this.maxRetries = MAX_RETRY_ATTEMPTS,
  });
}

// === 例外クラス（簡略化） ===
class AudioAnalysisException implements Exception {
  final String message;
  final dynamic originalError;
  
  AudioAnalysisException(this.message, [this.originalError]);
  
  @override
  String toString() => 'AudioAnalysisException: $message';
}

class AudioValidationException extends AudioAnalysisException {
  AudioValidationException(String message) : super(message);
}

class KaldiServiceException extends AudioAnalysisException {
  final int? statusCode;
  final DioExceptionType? dioErrorType;
  
  KaldiServiceException(
    String message, {
    this.statusCode,
    this.dioErrorType,
    dynamic originalError,
  }) : super(message, originalError);
  
  @override
  String toString() {
    final statusInfo = statusCode != null ? ' (Status: $statusCode)' : '';
    final typeInfo = dioErrorType != null ? ' (Type: $dioErrorType)' : '';
    return 'KaldiServiceException: $message$statusInfo$typeInfo';
  }
}

// === データクラス ===
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
    return AnalysisDetails(
      pronunciation: (json['pronunciation'] ?? 0.0).toDouble(),
      intonation: (json['intonation'] ?? 0.0).toDouble(),
      rhythm: (json['rhythm'] ?? 0.0).toDouble(),
    );
  }
}

class AnalysisResult {
  final bool success;
  final double? score;
  final String? feedback;
  final AnalysisDetails? details;
  final String? errorMessage;
  final Map<String, dynamic>? diagnostics;

  const AnalysisResult({
    required this.success,
    this.score,
    this.feedback,
    this.details,
    this.errorMessage,
    this.diagnostics,
  });

  factory AnalysisResult.error(String message, {Map<String, dynamic>? diagnostics}) {
    return AnalysisResult(
      success: false, 
      errorMessage: message,
      diagnostics: diagnostics,
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
      diagnostics: json['diagnostics'],
    );
  }
}

// === バリデーター ===
class AudioValidator {
  final int maxFileSizeBytes;

  AudioValidator({int maxFileSizeMB = MAX_FILE_SIZE_MB}) 
      : maxFileSizeBytes = maxFileSizeMB * 1024 * 1024;

  Future<void> validate(String audioFilePath) async {
    final file = File(audioFilePath);
    
    if (!await file.exists()) {
      throw AudioValidationException('Audio file does not exist: $audioFilePath');
    }

    final fileSize = await file.length();
    
    if (fileSize > maxFileSizeBytes) {
      final fileSizeMB = (fileSize / 1024 / 1024).toStringAsFixed(2);
      final maxSizeMB = (maxFileSizeBytes / 1024 / 1024).toStringAsFixed(0);
      throw AudioValidationException(
        'File size (${fileSizeMB}MB) exceeds limit (${maxSizeMB}MB)'
      );
    }

    if (fileSize == 0) {
      throw AudioValidationException('Audio file is empty');
    }

    print('✅ File validation passed: ${(fileSize / 1024).toStringAsFixed(2)}KB');
  }
}

// === ネットワーククライアント ===
class NetworkClient {
  late final Dio _dio;
  final AnalysisConfig config;

  NetworkClient(this.config) {
    _initializeDio();
  }

  void _initializeDio() {
    _dio = Dio(BaseOptions(
      connectTimeout: config.connectTimeout,
      receiveTimeout: config.receiveTimeout,
      sendTimeout: config.sendTimeout,
      headers: {
        'User-Agent': 'AudioAnalysisApp/1.0',
        'Accept': 'application/json',
      },
    ));

    if (kIsWeb) {
      _configureForWeb();
    }

    _dio.interceptors.addAll([
      _LoggingInterceptor(),
      _RetryInterceptor(config.maxRetries),
    ]);
  }

  void _configureForWeb() {
    (_dio.httpClientAdapter as DefaultHttpClientAdapter).onHttpClientCreate = (client) {
      client.badCertificateCallback = (cert, host, port) => true;
      return client;
    };
  }

  Future<AnalysisResult> sendAnalysisRequest(String audioFilePath) async {
    try {
      final formData = FormData.fromMap({
        'audio_file': await MultipartFile.fromFile(
          audioFilePath,
          filename: audioFilePath.split('/').last,
        ),
      });

      final response = await _dio.post(
        config.serviceUrl,
        data: formData,
        options: Options(
          headers: {'Content-Type': 'multipart/form-data'},
        ),
      );

      print('✅ API request successful (${response.statusCode})');
      return AnalysisResult.fromJson(response.data);
    } on DioException catch (e) {
      throw KaldiServiceException(
        _getErrorMessage(e),
        statusCode: e.response?.statusCode,
        dioErrorType: e.type,
        originalError: e.error,
      );
    }
  }

  Future<bool> testConnectivity() async {
    try {
      if (!kIsWeb) {
        final connectivityResult = await Connectivity().checkConnectivity();
        if (connectivityResult == ConnectivityResult.none) return false;
      }

      final uri = Uri.parse(config.serviceUrl);
      final healthUrl = '${uri.scheme}://${uri.host}:${uri.port}/health';
      
      await _dio.get(healthUrl, options: Options(
        receiveTimeout: Duration(seconds: 10),
      ));
      
      print('🌐 Network connectivity test passed');
      return true;
    } catch (e) {
      print('⚠️ Network connectivity test failed: $e');
      return false;
    }
  }

  String _getErrorMessage(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
        return 'Connection timeout after ${config.connectTimeout.inSeconds}s';
      case DioExceptionType.sendTimeout:
        return 'Upload timeout after ${config.sendTimeout.inSeconds}s';
      case DioExceptionType.receiveTimeout:
        return 'Response timeout after ${config.receiveTimeout.inSeconds}s';
      case DioExceptionType.badResponse:
        return 'Server error (${e.response?.statusCode}): ${e.response?.statusMessage}';
      case DioExceptionType.connectionError:
        return _getConnectionErrorMessage(e);
      case DioExceptionType.cancel:
        return 'Request was cancelled';
      case DioExceptionType.badCertificate:
        return 'SSL certificate validation failed';
      default:
        return 'Unknown network error occurred';
    }
  }

  String _getConnectionErrorMessage(DioException e) {
    if (kIsWeb && e.error?.toString().contains('XMLHttpRequest') == true) {
      return 'Web network error. Please check:\n'
             '• Server CORS configuration\n'
             '• Mixed content policy (HTTPS/HTTP)\n'
             '• Server availability\n'
             '• Browser console for detailed errors';
    }
    return 'Connection error. Check network and server availability.';
  }

  Future<Map<String, dynamic>> getNetworkDiagnostics() async {
    final diagnostics = <String, dynamic>{
      'timestamp': DateTime.now().toIso8601String(),
      'service_url': config.serviceUrl,
      'timeouts': {
        'connect': config.connectTimeout.inSeconds,
        'receive': config.receiveTimeout.inSeconds,
        'send': config.sendTimeout.inSeconds,
      },
    };

    try {
      final uri = Uri.parse(config.serviceUrl);
      final healthCheckUrl = '${uri.scheme}://${uri.host}:${uri.port}/health';
      
      final stopwatch = Stopwatch()..start();
      final response = await _dio.get(
        healthCheckUrl,
        options: Options(receiveTimeout: Duration(seconds: 10)),
      );
      stopwatch.stop();

      diagnostics['health_check'] = {
        'status': 'success',
        'status_code': response.statusCode,
        'response_time_ms': stopwatch.elapsedMilliseconds,
      };
    } catch (e) {
      diagnostics['health_check'] = {
        'status': 'failed',
        'error': e.toString(),
      };
    }

    return diagnostics;
  }

  void dispose() => _dio.close();
}

// === メインサービスクラス ===
class AnalyzeService {
  final AnalysisConfig config;
  final NetworkClient _networkClient;
  final AudioValidator _validator;
  final Map<String, AnalysisResult> _cache = {};

  // 既存コードとの互換性を保つためのコンストラクタ
  AnalyzeService({
    int maxFileSizeMB = MAX_FILE_SIZE_MB,
    Duration timeoutDuration = TIMEOUT_DURATION,
    String kaldiServiceUrl = KALDI_SERVICE_URL,
    AnalysisConfig? config,
  }) : config = config ?? AnalysisConfig(
          maxFileSizeMB: maxFileSizeMB,
          sendTimeout: timeoutDuration,
          serviceUrl: kaldiServiceUrl,
        ),
        _networkClient = NetworkClient(config ?? AnalysisConfig(
          maxFileSizeMB: maxFileSizeMB,
          sendTimeout: timeoutDuration,
          serviceUrl: kaldiServiceUrl,
        )),
        _validator = AudioValidator(maxFileSizeMB: 
          config?.maxFileSizeMB ?? maxFileSizeMB);

  Future<AnalysisResult> analyzeAudio(String audioFilePath) async {
    try {
      // キャッシュチェック
      if (_cache.containsKey(audioFilePath)) {
        print('📋 Using cached result for: $audioFilePath');
        return _cache[audioFilePath]!;
      }

      // ファイル検証
      await _validator.validate(audioFilePath);

      // ネットワーク接続テスト（必須ではない）
      await _networkClient.testConnectivity();

      // APIリクエスト
      final result = await _networkClient.sendAnalysisRequest(audioFilePath);

      // キャッシュに保存
      _cache[audioFilePath] = result;
      
      return result;
    } on AudioValidationException {
      rethrow;
    } on KaldiServiceException {
      rethrow;
    } catch (e) {
      throw AudioAnalysisException('Unexpected error during analysis', e);
    }
  }

  Future<Map<String, dynamic>> getNetworkDiagnostics() async {
    final diagnostics = await _networkClient.getNetworkDiagnostics();
    diagnostics['cache_size'] = _cache.length;
    return diagnostics;
  }

  void clearCache() {
    _cache.clear();
    print('🗑️ Analysis cache cleared');
  }

  void dispose() {
    _cache.clear();
    _networkClient.dispose();
  }
}

// === インターセプター（簡略化） ===
class _LoggingInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    if (kDebugMode) {
      print('🚀 Request: ${options.method} ${options.uri}');
    }
    handler.next(options);
  }

  @override
  void onResponse(Response response, ResponseInterceptorHandler handler) {
    if (kDebugMode) {
      print('✅ Response: ${response.statusCode}');
    }
    handler.next(response);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    print('❌ Network Error: ${err.type} - ${err.message}');
    handler.next(err);
  }
}

class _RetryInterceptor extends Interceptor {
  final int maxRetries;
  static const Duration retryDelay = Duration(seconds: 2);
  static const List<DioExceptionType> retryableErrors = [
    DioExceptionType.connectionTimeout,
    DioExceptionType.receiveTimeout,
    DioExceptionType.connectionError,
  ];

  _RetryInterceptor(this.maxRetries);

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    final retryCount = err.requestOptions.extra['retry_count'] ?? 0;

    if (retryCount < maxRetries && retryableErrors.contains(err.type)) {
      print('🔄 Retrying request (${retryCount + 1}/$maxRetries)...');
      
      await Future.delayed(retryDelay);
      
      final newOptions = err.requestOptions.copyWith(
        extra: {...err.requestOptions.extra, 'retry_count': retryCount + 1},
      );

      try {
        final response = await Dio().fetch(newOptions);
        handler.resolve(response);
        return;
      } catch (e) {
        // 再試行も失敗した場合は元のエラーを継続
      }
    }

    handler.next(err);
  }
}

// === パフォーマンス測定（既存コードとの互換性） ===
class RecordingPerformanceLogger {
  static final Map<String, DateTime> _startTimes = {};
  static final Map<String, List<Duration>> _measurements = {};

  static void start(String operation) {
    _startTimes[operation] = DateTime.now();
  }

  static void end(String operation) {
    final startTime = _startTimes[operation];
    if (startTime != null) {
      final duration = DateTime.now().difference(startTime);
      _measurements[operation] ??= [];
      _measurements[operation]!.add(duration);
      _startTimes.remove(operation);
      
      if (kDebugMode) {
        final durations = _measurements[operation]!;
        final avg = durations.reduce((a, b) => a + b).inMilliseconds / durations.length;
        final maxDuration = durations.map((d) => d.inMilliseconds).reduce(max);
        print('⏱️ $operation: ${duration.inMilliseconds}ms (avg=${avg.toStringAsFixed(1)}ms, max=${maxDuration}ms)');
      }
    }
  }

  static void logStats() {
    if (!kDebugMode) return;
    
    print('\n📊 Performance Stats:');
    _measurements.forEach((operation, durations) {
      if (durations.isNotEmpty) {
        final avg = durations.reduce((a, b) => a + b).inMilliseconds / durations.length;
        final maxDuration = durations.map((d) => d.inMilliseconds).reduce(max);
        final minDuration = durations.map((d) => d.inMilliseconds).reduce((a, b) => a < b ? a : b);
        print('  $operation: avg=${avg.toStringAsFixed(1)}ms, min=${minDuration}ms, max=${maxDuration}ms');
      }
    });
  }

  static void reset() {
    _startTimes.clear();
    _measurements.clear();
  }
}