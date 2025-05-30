import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:bold_chinese/screens/transaction_screen.dart';
// 今後実装
import 'package:bold_chinese/screens/phoneme_practice_screen.dart';
import 'package:bold_chinese/screens/history_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer(
      builder: (context, ref, child) {
        final size = MediaQuery.of(context).size;
        
        return Scaffold(
          appBar: AppBar(
            title: const Text(
              '今日もいつものルーティンお疲れ様です！\n一緒に中国語勉強頑張っていきましょ!加油!',
              style: TextStyle(fontSize: 18),
            ),
          ),
          body: SafeArea(
            child: Center(
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: 600),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    PracticeButton(
                      title: '自作文章で練習する',
                      onPressed: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const PracticeScreen()),
                      ),
                    ),
                    const SizedBox(height: 20),
                    PracticeButton(
                      title: '音ごとに練習する',
                      onPressed: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const PhonemePracticeScreen()),
                      ),
                    ),
                    const SizedBox(height: 20),
                    PracticeButton(
                      title: '履歴から提案された苦手な音を練習する',
                      onPressed: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const HistoryScreen()),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class PracticeButton extends StatelessWidget {
  final String title;
  final VoidCallback onPressed;

  const PracticeButton({
    required this.title,
    required this.onPressed,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          textStyle: const TextStyle(fontSize: 18),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        onPressed: onPressed,
        child: Text(title),
      ),
    );
  }
}