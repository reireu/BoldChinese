// lib/services/analyze_service.dart
import 'dart:async';
import 'dart:io';
import 'dart:convert'; // JSONエンコード/デコードのために必要
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;

/// 音声分析結果を表すモデルクラス。
/// バックエンドのAPIレスポンスのJSON構造と一致します。
class AnalysisResult {
  final bool success;
  final double? score; // null許容
  final String? feedback; // null許容
  final AnalysisDetails? details; // null許容
  final String? errorMessage; // エラーが発生した場合にメッセージを保持

  AnalysisResult({
    required this.success,
    this.score,
    this.feedback,
    this.details,
    this.errorMessage,
  });

  /// JSONマップからAnalysisResultオブジェクトを生成するファクトリコンストラクタ。
  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      success: json['success'] as bool,
      score: (json['score'] as num?)?.toDouble(),
      feedback: json['feedback'] as String?,
      details: json['details'] != null
          ? AnalysisDetails.fromJson(json['details'] as Map<String, dynamic>)
          : null,
      errorMessage: null, // 成功レスポンスなのでエラーメッセージはなし
    );
  }

  /// エラーが発生した場合のAnalysisResultオブジェクトを生成するファクトリコンストラクタ。
  factory AnalysisResult.withError(String message) {
    return AnalysisResult(
      success: false,
      errorMessage: message,
    );
  }
}

/// 分析結果の詳細スコアを表すモデルクラス。
class AnalysisDetails {
  final double? pronunciation;
  final double? intonation;
  final double? rhythm;

  AnalysisDetails({
    this.pronunciation,
    this.intonation,
    this.rhythm,
  });

  /// JSONマップからAnalysisDetailsオブジェクトを生成するファクトリコンストラクタ。
  factory AnalysisDetails.fromJson(Map<String, dynamic> json) {
    return AnalysisDetails(
      pronunciation: (json['pronunciation'] as num?)?.toDouble(),
      intonation: (json['intonation'] as num?)?.toDouble(),
      rhythm: (json['rhythm'] as num?)?.toDouble(),
    );
  }
}

/// 音声分析サービス。
/// バックエンドのKaldiサービスと連携し、音声ファイルのアップロードと分析結果の取得を行います。
class AnalyzeService {
  // KaldiサービスのエンドポイントURL。
  // 本番環境では、環境変数や設定ファイルから読み込むことを推奨します。
  static const String kaldiServiceUrl = 'http://localhost:8000/analyze'; // Pythonバックエンドのポート8000に合わせる

  /// 指定された音声ファイルをKaldiサービスに送信し、分析結果を取得します。
  ///
  /// [audioFile] 分析する音声ファイル（Fileオブジェクト）。
  /// 戻り値: [AnalysisResult] オブジェクト。成功またはエラー情報を含みます。
  Future<AnalysisResult> analyzeAudio(File audioFile) async {
    try {
      // 1. ファイル形式の検証
      if (!_isValidAudioFile(audioFile)) {
        return AnalysisResult.withError('無効な音声ファイル形式です。WAVまたはMP3ファイルのみが許可されています。');
      }

      // 2. Kaldiサービスへのリクエスト送信
      final http.Response response = await _sendToKaldiService(audioFile);

      // 3. HTTPレスポンスのステータスコードをチェック
      if (response.statusCode == 200) {
        // 成功レスポンスの場合
        final Map<String, dynamic> jsonResponse = json.decode(response.body);
        return AnalysisResult.fromJson(jsonResponse);
      } else {
        // エラーレスポンスの場合
        String errorMessage;
        try {
          final Map<String, dynamic> errorJson = json.decode(response.body);
          // バックエンドからの詳細なエラーメッセージがあればそれを使用
          errorMessage = errorJson['detail'] as String? ?? '不明なエラーが発生しました。';
        } catch (_) {
          // JSONパースに失敗した場合
          errorMessage = 'サーバーからのエラーレスポンスを解析できませんでした。ステータスコード: ${response.statusCode}';
        }

        switch (response.statusCode) {
          case 400:
            return AnalysisResult.withError('リクエストが無効です: $errorMessage');
          case 500:
            return AnalysisResult.withError('サーバー内部エラーが発生しました: $errorMessage');
          case 504:
            return AnalysisResult.withError('音声分析サービスがタイムアウトしました: $errorMessage');
          default:
            return AnalysisResult.withError('予期せぬエラーが発生しました。ステータスコード: ${response.statusCode}, メッセージ: $errorMessage');
        }
      }
    } on SocketException {
      // ネットワーク接続エラー
      return AnalysisResult.withError('ネットワーク接続に失敗しました。サーバーが起動しているか確認してください。');
    } on TimeoutException {
      // リクエストのタイムアウト（Dartのhttpクライアント側）
      return AnalysisResult.withError('サーバーへの接続がタイムアウトしました。');
    } catch (e) {
      // その他の予期せぬエラー
      return AnalysisResult.withError('音声分析中に予期せぬエラーが発生しました: ${e.toString()}');
    }
  }

  /// 指定されたファイルが有効な音声ファイル形式（.wav または .mp3）であるかを検証します。
  ///
  /// [file] 検証するFileオブジェクト。
  /// 戻り値: 有効な形式であればtrue、そうでなければfalse。
  bool _isValidAudioFile(File file) {
    final extension = path.extension(file.path).toLowerCase();
    return ['.wav', '.mp3'].contains(extension);
  }

  /// 音声ファイルをKaldiサービスにMultipartリクエストとして送信します。
  ///
  /// [audioFile] 送信する音声ファイル。
  /// 戻り値: サーバーからの[http.Response]オブジェクト。
  /// 例外: 通信エラーが発生した場合にExceptionをスローします。
  Future<http.Response> _sendToKaldiService(File audioFile) async {
    // MultipartRequestを作成
    var request = http.MultipartRequest('POST', Uri.parse(kaldiServiceUrl));

    // 音声ファイルをリクエストに追加
    request.files.add(
      await http.MultipartFile.fromPath(
        'audio_file', // バックエンドのFastAPIエンドポイントで定義されたフィールド名と一致させる
        audioFile.path,
        filename: path.basename(audioFile.path), // ファイル名を指定
      ),
    );

    // リクエストを送信し、ストリームされたレスポンスを取得
    final streamedResponse = await request.send();

    // ストリームされたレスポンスを完全なHTTPレスポンスに変換
    return await http.Response.fromStream(streamedResponse);
  }

  // 以下は、将来的にバックエンドから詳細な分析結果を受け取った際に、
  // フロントエンド側で追加の加工や表示ロジックを実装するためのプレースホルダーです。
  // 現在は、バックエンドが最終的なスコアとフィードバックを返すため、
  // これらのメソッドは直接使用されないか、AnalysisResultモデルから値を取得する形になります。

  /// バックエンドからのレスポンスに基づいて全体スコアを計算（または取得）します。
  ///
  /// [response] バックエンドからの解析済みJSONレスポンス。
  /// 戻り値: 計算された（または取得された）スコア。
  double _calculateScore(Map<String, dynamic> response) {
    // 実際にはresponse['score']から値を取得します
    return (response['score'] as num?)?.toDouble() ?? 0.0;
  }

  /// バックエンドからのレスポンスに基づいてフィードバックを生成（または取得）します。
  ///
  /// [response] バックエンドからの解析済みJSONレスポンス。
  /// 戻り値: 生成された（または取得された）フィードバック文字列。
  String _generateFeedback(Map<String, dynamic> response) {
    // 実際にはresponse['feedback']から値を取得します
    return response['feedback'] as String? ?? "フィードバックがありません。";
  }

  /// バックエンドからのレスポンスに基づいて発音スコアを計算（または取得）します。
  ///
  /// [response] バックエンドからの解析済みJSONレスポンス。
  /// 戻り値: 計算された（または取得された）発音スコア。
  double _calculatePronunciationScore(Map<String, dynamic> response) {
    // 実際にはresponse['details']['pronunciation']から値を取得します
    return (response['details']?['pronunciation'] as num?)?.toDouble() ?? 0.0;
  }

  /// バックエンドからのレスポンスに基づいてイントネーションスコアを計算（または取得）します。
  ///
  /// [response] バックエンドからの解析済みJSONレスポンス。
  /// 戻り値: 計算された（または取得された）イントネーションスコア。
  double _calculateIntonationScore(Map<String, dynamic> response) {
    // 実際にはresponse['details']['intonation']から値を取得します
    return (response['details']?['intonation'] as num?)?.toDouble() ?? 0.0;
  }

  /// バックエンドからのレスポンスに基づいてリズムスコアを計算（または取得）します。
  ///
  /// [response] バックエンドからの解析済みJSONレスポンス。
  /// 戻り値: 計算された（または取得された）リズムスコア。
  double _calculateRhythmScore(Map<String, dynamic> response) {
    // 実際にはresponse['details']['rhythm']から値を取得します
    return (response['details']?['rhythm'] as num?)?.toDouble() ?? 0.0;
  }
}