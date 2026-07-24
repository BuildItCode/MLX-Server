# Performance (flutter/skills)

Source: flutter/skills (Apache 2.0), Flutter DevTools docs

---

## Widget Rebuild Minimisation

- `const` widgets everywhere possible — Flutter's primary optimisation
- Extract subtrees that never depend on changing state into `const` widgets
- Split large widgets so `@Observable`/`ChangeNotifier` granularity only rebuilds the consumer
- `RepaintBoundary` around independently animated or frequently-repainting widgets
- `AutomaticKeepAliveClientMixin` only for tab content genuinely expensive to rebuild

---

## Lists

- `ListView.builder` / `GridView.builder` — never `ListView(children: list.map(...))`
- `SliverList(delegate: SliverChildBuilderDelegate(...))` for custom scroll effects
- Stable keys for list items — identity changes cause full cell recreation

---

## Images

- `cached_network_image` with explicit `width`, `height`, `fit`, `memCacheWidth`
- Downsample before display: use `ImageCacheManager` or `ResizeImage`
- Never call `Image.network` inline without cache or size constraints

---

## Isolates

- `compute(fn, message)` for heavy Dart work (JSON parsing > ~5ms, image processing)
- Never block the main isolate with synchronous heavy computation
- `Isolate.spawn` for long-running background tasks

---

## Build Method Rules

- No expensive operations in `build()` — precompute in notifiers or `didChangeDependencies`
- No `Future.wait` or `async` calls inside `build()` — use `.task` modifier equivalent (FutureBuilder/StreamBuilder)
- Cache expensive derived values with `late final` or `_cachedValue ??= compute()`

---

## Profiling

- Flutter DevTools **Performance** tab: frame timeline, jank detection
- Flutter DevTools **Widget Rebuilds**: identify hot widgets
- `flutter run --profile` — never `debug` for performance profiling
- Enable **Paint** and **Repaint Rainbow** in DevTools to spot excess repaints
- `debugPrintRebuildDirtyWidgets = true` during development to trace rebuild storms

---

## App Size (flutter/skills — bundle-size skill)

```bash
# Analyze release build size
flutter build apk --analyze-size
flutter build appbundle --analyze-size
flutter build ios --analyze-size

# Deferred loading for large features
import 'package:deferred_widget/deferred_widget.dart' deferred as feature;
await feature.loadLibrary();
```

- Enable R8/ProGuard on Android: `minifyEnabled = true`
- `--obfuscate --split-debug-info=<dir>` for release builds
- Tree-shaking is automatic — avoid `dart:mirrors`
