import 'dart:async';
import 'dart:io';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class KaldiService {
  final FlutterSoundRecorder _audioRecorder = FlutterSoundRecorder();
  bool _isRecorderInitialized = false;
  bool _isRecording = false;
  String? _recordingPath;

  Future<void> initialize() async {
    try {
      await _audioRecorder.openRecorder();
      _isRecorderInitialized = true;
    } catch (e) {
      print('Recorder initialization failed: $e');
    }
  }

  Future<void> startRecording() async {
    if (!_isRecorderInitialized || _isRecording) return;

    try {
      Directory tempDir = await getTemporaryDirectory();
      _recordingPath = '${tempDir.path}/recorded_audio.wav';
      await _audioRecorder.startRecorder(
        toFile: _recordingPath,
        codec: Codec.pcm16WAV,
      );
      _isRecording = true;
    } catch (e) {
      print('Start recording failed: $e');
    }
  }

  Future<String?> stopRecording() async {
    if (!_isRecorderInitialized || !_isRecording) return null;

    try {
      await _audioRecorder.stopRecorder();
      _isRecording = false;
      return _recordingPath;
    } catch (e) {
      print('Stop recording failed: $e');
      return null;
    }
  }

  Future<Map<String, dynamic>> analyzePronunciation(String audioFilePath, String text) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('https://your-kaldi-server.com/analyze'),
      );
      request.files.add(await http.MultipartFile.fromPath('audio', audioFilePath));
      request.fields['text'] = text;

      var response = await request.send();
      if (response.statusCode == 200) {
        var responseBody = await response.stream.bytesToString();
        return json.decode(responseBody);
      } else {
        print('Kaldi server error: ${response.statusCode}');
        return {'score': 0.0, 'details': []};
      }
    } catch (e) {
      print('Analyze pronunciation failed: $e');
      return {'score': 0.0, 'details': []};
    }
  }

  void dispose() {
    _audioRecorder.closeRecorder();
    _isRecorderInitialized = false;
  }
}
