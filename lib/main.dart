import 'package:flutter/material.dart';
import 'core/theme/app_theme.dart';
import 'core/routes/app_router.dart';
import 'services/kaldi_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Kaldiサービスの初期化
  await KaldiService.initialize();

  runApp(const BoldVoiceApp());
}

class BoldVoiceApp extends StatelessWidget {
  const BoldVoiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'BoldChinese',
      theme: AppTheme.lightTheme,
      routerConfig: AppRouter.router,
      debugShowCheckedModeBanner: false,
    );
  }
}
