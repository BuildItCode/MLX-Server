# Architecture — MVVM + Clean Layers

---

## Layer Map

```
UI          →  @Composable functions, UiState data classes
ViewModel   →  StateFlow<UiState>, event handlers, no Android/Compose types
Domain      →  Use cases (required when logic crosses ViewModel boundaries)
Data        →  Repository implementations, DataSources, DTOs + mappers
```

## UiState Pattern

```kotlin
// Sealed interface — allows multi-interface implementation
sealed interface LoginUiState {
    data object Idle : LoginUiState
    data object Loading : LoginUiState
    data class Success(val user: User) : LoginUiState
    data class Error(val message: String, val cause: Throwable? = null) : LoginUiState
}
```

- ViewModel exposes a single `StateFlow<UiState>` per screen — not individual flows per field
- ViewModel never holds a reference to `Context`, `View`, `Activity`, or any Compose type
- Repositories are interfaces in domain; implementations live in data
- Use cases are `operator fun invoke()` classes with no state

## One-Shot Events

```kotlin
// ✓ Channel + receiveAsFlow for navigation and one-shot UI events
private val _events = Channel<LoginEvent>(Channel.BUFFERED)
val events = _events.receiveAsFlow()

// ✗ StateFlow for events — redelivered on resubscription
private val _navigateToHome = MutableStateFlow(false)
```

## Dependency Injection — Hilt

- Every ViewModel annotated `@HiltViewModel`; every entry point `@AndroidEntryPoint`
- Scopes: `@Singleton` app-wide; `@ActivityRetainedScoped` ViewModel-scoped; `@ViewModelScoped` internal
- No `object` singletons bypassing Hilt
- Test modules use `@TestInstallIn`

## Testing

| Layer | Tool | Scope |
|---|---|---|
| Unit | JUnit 5 + MockK | ViewModels, use cases, repositories, mappers |
| Compose UI | `compose-ui-test` | Individual composables — not full screens |
| Integration | Hilt + `AndroidComposeTestRule` | Feature flows |
| E2E | Maestro | Critical journeys |

- `Turbine` for `StateFlow`/`Flow` assertions
- `UnconfinedTestDispatcher` for coroutine tests
- MockK over Mockito — coroutine-aware, supports `suspend` mocking
