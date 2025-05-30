import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:bold_chinese/screens/home_screen.dart';
import 'package:bold_chinese/services/analyze_service.dart';

// グローバルプロバイダーの設定
final analyzeServiceProvider = Provider((ref) => AnalyzeService());

void main() {
  runApp(
    const ProviderScope(
      child: MyApp(),
    ),
  );
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