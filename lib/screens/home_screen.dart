import 'package:flutter/material.dart';
import 'package:bold_chinese/screens/transaction_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          '今日もいつものルーティンお疲れ様です！\n一緒に中国語勉強頑張っていきましょ!加油!',
          style: TextStyle(fontSize: 18),
        ),
      ),
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const TranscriptionScreen(),
                      ),
                    );
                  },
                  child: const Text('発音練習を始める'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}