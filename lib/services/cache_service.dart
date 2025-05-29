// lib/services/cache_service.dart

class CacheService {
  final _cache = <String, AnalysisResult>{};
  final Duration _maxAge = Duration(hours: 24);
  
  Future<AnalysisResult?> get(String key) async {
    final cached = _cache[key];
    if (cached != null) {
      // キャッシュの有効期限チェック
      return cached;
    }
    return null;
  }
  
  Future<void> set(String key, AnalysisResult value) async {
    _cache[key] = value;
  }
}