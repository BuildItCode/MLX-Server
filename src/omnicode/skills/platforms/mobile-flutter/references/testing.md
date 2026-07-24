# Testing (flutter/skills)

Source: flutter/skills (Apache 2.0)

---

## Test Pyramid (flutter/skills)

| Layer | Tool | Scope |
|---|---|---|
| Unit | `test` package | ViewModels, use cases, repositories, services, utils |
| Widget | `flutter_test` / `WidgetTester` | Individual widgets — not full screens |
| Integration | `integration_test` | Critical user flows on device/emulator |
| Golden | `golden_toolkit` | Design-critical widgets |
| Mocking | `mocktail` or **Fakes** | External dependencies |

**Fakes over Mocks** (flutter/skills rule): Prefer `FakeUserRepository` implementations over `MockRepository.when()` — fakes provide well-defined inputs/outputs and test ViewModel/View without implementation coupling.

---

## Checklist (flutter/skills)

For each new architectural feature:
1. Create `Fake` implementations for any new Repositories or Services
2. Write unit tests for the Repository (mocking the API/Database client)
3. Write unit tests for the ViewModel (injecting Fake Repositories)
4. Write widget tests for the View (injecting ViewModel and Fake Repositories)
5. Write an integration test for the critical path
6. Run `flutter test --coverage` → review gaps → fix missing edge cases

---

## Unit Tests

```dart
void main() {
  group('UserDetailViewModel', () {
    late UserDetailViewModel viewModel;
    late FakeUserRepository repository;

    setUp(() {
      repository = FakeUserRepository()..addUser(kTestUser);
      viewModel = UserDetailViewModel(repository: repository);
    });

    test('loads user successfully', () async {
      await viewModel.load('test-id');
      expect(viewModel.user, equals(kTestUser));
    });

    test('sets error state on failure', () async {
      repository.shouldFail = true;
      await viewModel.load('test-id');
      expect(viewModel.error, isNotNull);
    });
  });
}
```

---

## Widget Tests (flutter/skills pattern)

```dart
testWidgets('renders bookings list', (WidgetTester tester) async {
  final repository = FakeBookingRepository()..createBooking(kBooking);
  final viewModel = HomeViewModel(bookingRepository: repository);

  await tester.pumpWidget(
    MaterialApp(home: HomeScreen(viewModel: viewModel)),
  );

  expect(find.byType(ListView), findsOneWidget);
  expect(find.text('Booking 1'), findsOneWidget);
});
```

- Pump the smallest widget that exercises the behaviour
- `pumpAndSettle` only when animations are involved — `pump(duration)` for control
- Assert on user-visible text and widget types — not internal state

---

## Integration Tests (flutter/skills)

```dart
// integration_test/app_test.dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('counter increments', (tester) async {
    await tester.pumpWidget(const MyApp());
    expect(find.text('0'), findsOneWidget);
    await tester.tap(find.byKey(const ValueKey('increment')));
    await tester.pumpAndSettle();
    expect(find.text('1'), findsOneWidget);
  });
}
```

Run on device/emulator:
```bash
flutter drive --driver=test_driver/integration_test.dart \
              --target=integration_test/app_test.dart
```

---

## Plugin Testing (flutter/skills)

- Dart unit tests: mock the platform channel to validate Dart logic
- Native unit tests: isolated platform logic on Android/iOS side
- Combine both for full method-channel coverage

---

## BLoC Tests

```dart
blocTest<UserDetailCubit, UserDetailState>(
  'emits success after load',
  build: () => UserDetailCubit(FakeUserRepository()..addUser(kTestUser)),
  act: (cubit) => cubit.load('test-id'),
  expect: () => [const UserDetailSuccess(kTestUser)],
);
```
