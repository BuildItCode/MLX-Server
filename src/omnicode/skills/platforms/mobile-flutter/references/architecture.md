# Architecture (flutter/skills — MVVM + Layered)

Source: flutter/skills (Apache 2.0), flutter.dev/app-architecture

---

## Layer Rules (flutter/skills strict)

```
UI Layer        →  Widgets + ViewModels (ChangeNotifier or Cubit/Notifier)
Domain Layer    →  UseCases (optional — only when logic spans multiple repositories)
Data Layer      →  Repositories (single source of truth) + Services (stateless API wrappers)
```

**No Logic in Views**: Views contain only layout logic, simple conditional rendering on ViewModel state, and routing calls. Zero business logic.

**Unidirectional Data Flow**: Data flows Data Layer → UI Layer only. UI events trigger ViewModel commands; commands call repositories.

**Single Source of Truth**: Repositories are the only classes that mutate application data.

**Service Isolation**: ViewModels never call Services directly — only through Repositories or UseCases.

**Stateless Services**: Service classes have no state. Their sole responsibility is wrapping external APIs or local storage.

**Immutable Models**: Domain models passed from Repositories to ViewModels are immutable (`final` fields, `const` constructors, or `freezed`).

---

## Decision Logic (flutter/skills)

```
Feature requires external API?       → Create a Remote Service
Feature requires local storage?      → Create a Local Service
Logic shared across ViewModels?      → Implement UseCase in Domain Layer
Standard CRUD / simple display?      → Skip Domain Layer; ViewModel → Repository directly
```

---

## Data Layer

### Service (stateless API wrapper)
```dart
class SharedPreferencesService {
  static const String _kDarkMode = 'darkMode';

  Future<void> setDarkMode(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kDarkMode, value);
  }

  Future<bool> isDarkMode() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_kDarkMode) ?? false;
  }
}
```

### Repository (source of truth, Result types)
```dart
class SettingsRepository {
  final SharedPreferencesService _service;
  SettingsRepository(this._service);

  Future<Result<bool>> isDarkMode() async {
    try {
      return Result.ok(await _service.isDarkMode());
    } catch (e) {
      return Result.error(Exception('Failed to read dark mode: $e'));
    }
  }
}
```

---

## UI Layer

### ViewModel (ChangeNotifier — flutter/skills default)
```dart
class SettingsViewModel extends ChangeNotifier {
  final SettingsRepository _repository;
  SettingsViewModel(this._repository);

  bool _isDarkMode = false;
  bool get isDarkMode => _isDarkMode;

  late final loadSettings = Command0(_loadSettings);

  Future<Result<void>> _loadSettings() async {
    final result = await _repository.isDarkMode();
    if (result is Ok<bool>) {
      _isDarkMode = result.value;
      notifyListeners();
    }
    return result;
  }
}
```

### View (no logic — only ListenableBuilder)
```dart
class SettingsScreen extends StatelessWidget {
  final SettingsViewModel viewModel;
  const SettingsScreen({required this.viewModel, super.key});

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: viewModel,
    builder: (context, _) => Switch(
      value: viewModel.isDarkMode,
      onChanged: (_) => viewModel.loadSettings.execute(),
    ),
  );
}
```

---

## Folder Structure (feature-first)

```
lib/
  features/
    {feature}/
      data/
        models/         DTOs (fromJson/toJson)
        datasources/    remote + local source impls
        repositories/   repository implementations
      domain/
        entities/       pure Dart models (no serialisation)
        repositories/   abstract interfaces
        usecases/       single-method use cases (optional)
      presentation/
        screens/        route entry points — wiring only
        widgets/        feature widgets
        viewmodels/     ChangeNotifier or Cubit/Notifier
  core/
    theme/
    routing/
    network/
    di/
    utils/
```
