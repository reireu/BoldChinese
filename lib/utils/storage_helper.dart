import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:localstorage/localstorage.dart';
import 'dart:io';
import 'package:path/path.dart' as path;

class StorageHelper {
  static final storage = LocalStorage('app_storage');

  static Future<String> getTemporaryPath() async {
    if (kIsWeb) {
      await storage.ready;
      return storage.getItem('temp_path') ?? '';
    } else {
      final dir = await getTemporaryDirectory();
      return dir.path;
    }
  }

  static Future<void> saveData(String key, dynamic value) async {
    try {
      if (kIsWeb) {
        await storage.ready;
        await storage.setItem(key, value);
      } else {
        final dir = await getTemporaryDirectory();
        final filePath = path.join(dir.path, key);
        final file = await File(filePath).create(recursive: true);
        await file.writeAsString(value.toString());
      }
    } catch (e) {
      print('Storage error: $e');
      rethrow;
    }
  }

  static Future<String?> getData(String key) async {
    try {
      if (kIsWeb) {
        await storage.ready;
        return storage.getItem(key)?.toString();
      } else {
        final dir = await getTemporaryDirectory();
        final filePath = path.join(dir.path, key);
        final file = File(filePath);
        if (await file.exists()) {
          return await file.readAsString();
        }
      }
      return null;
    } catch (e) {
      print('Read error: $e');
      return null;
    }
  }
}