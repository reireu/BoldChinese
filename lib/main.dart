import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:bold_chinese/screens/home_screen.dart';
import 'package:bold_chinese/services/analyze_service.dart';
import 'package:flutter/foundation.dart';
import 'package:localstorage/localstorage.dart';

// グローバルプロバイダーの設定
final analyzeServiceProvider = Provider((ref) => AnalyzeService(
  maxFileSizeMB: MAX_FILE_SIZE_MB,
  timeoutDuration: TIMEOUT_DURATION,
  kaldiServiceUrl: KALDI_SERVICE_URL,
));

void main() async {
  try {
    // Flutter bindings初期化
    WidgetsFlutterBinding.ensureInitialized();

    // Webプラットフォームの場合の初期化
    if (kIsWeb) {
      final storage = LocalStorage('bold_chinese');
      await storage.ready;
    }

    runApp(
      const ProviderScope(
        child: MyApp(),
      ),
    );
  } catch (e) {
    print('Initialization error: $e');
    // エラー時のフォールバックUI表示
    runApp(
      MaterialApp(
        home: Scaffold(
          body: Center(
            child: Text('アプリケーションの初期化中にエラーが発生しました。'),
          ),
        ),
      ),
    );
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BoldChinese',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFF8674A1),
        colorScheme: ColorScheme.fromSwatch().copyWith(
          secondary: const Color(0xFF8674A1),
        ),
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF8674A1),
          foregroundColor: Colors.white,
          elevation: 0,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}