import 'package:flutter/material.dart';
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_widgets.dart';

class AudioInputScreen extends StatefulWidget {
  const AudioInputScreen({Key? key}) : super(key: key);

  @override
  State<AudioInputScreen> createState() => _AudioInputScreenState();
}

class _AudioInputScreenState extends State<AudioInputScreen> {
  final TextEditingController _textController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

@override
void initState() {
  super.initState();
  _textController.addListener(() {
    setState(() {});
  });
}

  void _onConfirm() {
    if (_formKey.currentState?.validate() ?? false) {
      final userText = _textController.text.trim();
      // 次の画面（録音画面）に文章を渡して遷移
      Navigator.pushNamed(
        context, 
        '/transcription', 
        arguments: userText,
      );
    }
  }

  void _onClear() {
    _textController.clear();
  }

  @override
  Widget build(BuildContext context) {
    final theme = FlutterFlowTheme.of(context);

    return Scaffold(
      backgroundColor: const Color(0xFF8674A1),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ヘッダー部分
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(
                        Icons.arrow_back,
                        color: Colors.white,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '発音練習',
                      style: theme.headlineSmall.override(
                        fontFamily: 'Manrope',
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.0,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 32),
                
                // メインコンテンツカード
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(32),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // タイトル
                          Center(
                            child: Column(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(16),
                                  decoration: BoxDecoration(
                                    color: const Color(0xFF8674A1).withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                  child: const Icon(
                                    Icons.edit_note,
                                    color: Color(0xFF8674A1),
                                    size: 32,
                                  ),
                                ),
                                const SizedBox(height: 16),
                                Text(
                                  '練習したい文章を入力',
                                  style: theme.headlineSmall.override(
                                    fontFamily: 'Manrope',
                                    color: const Color(0xFF333333),
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.0,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  '発音練習したい文章をここに入力または\n貼り付けてください',
                                  style: theme.bodyMedium.override(
                                    fontFamily: 'Manrope',
                                    color: Colors.grey[600],
                                    letterSpacing: 0.0,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 32),
                          
                          // 入力フィールド
                          Text(
                            '文章',
                            style: theme.labelLarge.override(
                              fontFamily: 'Manrope',
                              color: const Color(0xFF333333),
                              fontWeight: FontWeight.w600,
                              letterSpacing: 0.0,
                            ),
                          ),
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _textController,
                            minLines: 6,
                            maxLines: 10,
                            decoration: InputDecoration(
                              hintText: '例: 私は今日、新しいプログラミング言語を学び始めました。この言語はとても興味深く、様々な可能性を秘めています。',
                              hintStyle: theme.bodyMedium.override(
                                fontFamily: 'Manrope',
                                color: Colors.grey[400],
                                letterSpacing: 0.0,
                              ),
                              filled: true,
                              fillColor: Colors.grey[50],
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: BorderSide(
                                  color: Colors.grey[300]!,
                                  width: 1.5,
                                ),
                              ),
                              enabledBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: BorderSide(
                                  color: Colors.grey[300]!,
                                  width: 1.5,
                                ),
                              ),
                              focusedBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: const BorderSide(
                                  color: Color(0xFF8674A1),
                                  width: 2,
                                ),
                              ),
                              errorBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: const BorderSide(
                                  color: Colors.red,
                                  width: 2,
                                ),
                              ),
                              focusedErrorBorder: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(16),
                                borderSide: const BorderSide(
                                  color: Colors.red,
                                  width: 2,
                                ),
                              ),
                              contentPadding: const EdgeInsets.symmetric(
                                vertical: 20,
                                horizontal: 16,
                              ),
                            ),
                            style: theme.bodyLarge.override(
                              fontFamily: 'Manrope',
                              color: const Color(0xFF333333),
                              letterSpacing: 0.0,
                              height: 1.5,
                            ),
                            validator: (value) {
                              if (value == null || value.trim().isEmpty) {
                                return '文章を入力してください';
                              }
                              if (value.trim().length < 10) {
                                return '10文字以上の文章を入力してください';
                              }
                              return null;
                            },
                          ),
                          const SizedBox(height: 16),
                          
                          // 文字数カウンター
                          Align(
                            alignment: Alignment.centerRight,
                            child: Text(
                              '${_textController.text.length} 文字',
                              style: theme.bodySmall.override(
                                fontFamily: 'Manrope',
                                color: Colors.grey[500],
                                letterSpacing: 0.0,
                              ),
                            ),
                          ),
                          const SizedBox(height: 32),
                          
                          // ボタン群
                          Row(
                            children: [
                              // クリアボタン
                              Expanded(
                                flex: 1,
                                child: FFButtonWidget(
                                  onPressed: _onClear,
                                  text: 'クリア',
                                  options: FFButtonOptions(
                                    height: 56,
                                    color: Colors.grey[100],
                                    textStyle: theme.titleMedium.override(
                                      fontFamily: 'Manrope',
                                      color: Colors.grey[700],
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 0.0,
                                    ),
                                    elevation: 0,
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                      width: 1,
                                    ),
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 16),
                              // 確定ボタン
                              Expanded(
                                flex: 2,
                                child: FFButtonWidget(
                                  onPressed: _onConfirm,
                                  text: '録音画面へ進む',
                                  options: FFButtonOptions(
                                    height: 56,
                                    color: const Color(0xFF8674A1),
                                    textStyle: theme.titleMedium.override(
                                      fontFamily: 'Manrope',
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                      letterSpacing: 0.0,
                                    ),
                                    elevation: 0,
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                
                // ヒント部分
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.white.withOpacity(0.2),
                      width: 1,
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.lightbulb_outline,
                        color: Colors.white.withOpacity(0.8),
                        size: 24,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          '効果的な練習のために、10文字以上の文章を入力することをお勧めします',
                          style: theme.bodySmall.override(
                            fontFamily: 'Manrope',
                            color: Colors.white.withOpacity(0.9),
                            letterSpacing: 0.0,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
