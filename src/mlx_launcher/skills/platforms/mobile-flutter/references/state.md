# State Management

Source: flutter/skills (Apache 2.0), Riverpod docs, BLoC docs

---

## Choosing a Solution

Declared in `ARCH.md` — do not switch mid-project.

| Pattern | Use when |
|---|---|
| **Riverpod** | New projects; code-gen with `riverpod_generator` |
| **BLoC/Cubit** | Team familiar with BLoC; complex event→state rules |
| **ChangeNotifier + Provider** | flutter/skills MVVM default; simpler apps |
| **setState** | Ephemeral local UI state only — not shared |

---

## Riverpod (New Projects)

```dart
// ✓ providers at file top level — never inside widgets
@riverpod
class UserDetail extends _$UserDetail {
  @override
  Future<User> build(String userId) =>
      ref.read(userRepositoryProvider).getById(userId);

  Future<void> refresh() async {
    ref.invalidateSelf();
    await future;
  }
}

// ✓ ConsumerWidget — screen owns no state
class UserDetailScreen extends ConsumerWidget {
  const UserDetailScreen({required this.userId, super.key});
  final String userId;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(userDetailProvider(userId));
    return switch (state) {
      AsyncLoading() => const CircularProgressIndicator(),
      AsyncError(:final error) => ErrorView(error: error),
      AsyncData(:final value) => UserDetailView(user: value),
    };
  }
}
```

**Rules:**
- All providers at file top level — never inside `build()` or widget constructors
- `ref.watch` in `build()`; `ref.read` only in callbacks and notifier methods
- `AsyncNotifierProvider` for async state; `NotifierProvider` for sync
- Notifiers have no Flutter or material library imports

---

## BLoC / Cubit

```dart
// State as sealed class (Dart 3+)
sealed class UserDetailState { const UserDetailState(); }
class UserDetailLoading extends UserDetailState { const UserDetailLoading(); }
class UserDetailSuccess extends UserDetailState {
  const UserDetailSuccess(this.user);
  final User user;
}
class UserDetailError extends UserDetailState {
  const UserDetailError(this.message);
  final String message;
}

// Cubit — no Flutter imports
class UserDetailCubit extends Cubit<UserDetailState> {
  UserDetailCubit(this._repository) : super(const UserDetailLoading());
  final UserRepository _repository;

  Future<void> load(String id) async {
    try {
      emit(UserDetailSuccess(await _repository.getById(id)));
    } on AppException catch (e) {
      emit(UserDetailError(e.message));
    }
  }
}

// Screen wires bloc — never StatefulWidget for route state
class UserDetailScreen extends StatelessWidget {
  const UserDetailScreen({required this.userId, super.key});
  final String userId;

  @override
  Widget build(BuildContext context) => BlocProvider(
    create: (ctx) => ctx.read<UserDetailCubit>()..load(userId),
    child: const _UserDetailBody(),
  );
}
```

**Rules:**
- `Cubit` for linear state; `Bloc` when event→state mapping has branching rules
- State classes use `sealed class` (Dart 3+)
- Cubits/Blocs have zero Flutter or material imports
- `BlocBuilder` renders; `BlocListener` for side effects; `BlocConsumer` only when both needed

---

## ChangeNotifier + Provider (flutter/skills MVVM default)

- ViewModel extends `ChangeNotifier`
- View uses `ListenableBuilder` (not `Consumer` or `context.watch`)
- ViewModel calls only repositories — never services directly
- `notifyListeners()` called after state mutation, never speculatively
