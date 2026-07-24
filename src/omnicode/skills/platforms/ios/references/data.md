# Data Flow & State

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple developer documentation

---

## Property Wrapper Selection

| Situation | Use |
|---|---|
| Local UI state (toggle, text field) | `@State` |
| Owned ViewModel / shared model | `@State` (with `@Observable`) |
| Injected `@Observable` object | plain stored property or `@Bindable` for two-way binding |
| App-wide injection | `@Environment(MyService.self)` |
| System environment values | `@Environment(\.colorScheme)` etc |
| Two-way binding into `@Observable` | `@Bindable` |
| Pre-iOS 17 shared data | `@StateObject` (owns) / `@ObservedObject` (injected) |
| Never for new code | `@EnvironmentObject`, `ObservableObject`, `@Published` |

---

## @Observable (iOS 17+) — Always for New Code

```swift
// ✓ correct — @Observable + @MainActor
@Observable @MainActor
final class UserViewModel {
    var name = ""
    var isLoading = false
    var error: AppError?

    func load(id: String) async { ... }
}

// In the view — @State for ownership
struct UserDetailView: View {
    @State private var viewModel = UserViewModel()
    var body: some View { ... }
}

// Passing in — plain property
struct UserCard: View {
    var viewModel: UserViewModel  // not @ObservedObject
    var body: some View { ... }
}
```

- Every `@Observable` class must be `@MainActor` unless the project uses whole-module actor isolation
- `objectWillChange.send()` never needed — `@Observable` tracks property access automatically
- `@Observable` is not retroactively available below iOS 17 — use `#if available` or `ObservableObject` for older targets

---

## State Ownership Rules

- A `View` that creates a model **owns** it: `@State private var model = MyModel()`
- A `View` that receives a model **does not own** it: plain `var model: MyModel`
- Never pass `@State` binding into a child that doesn't need to write back — pass the value instead
- `@Binding` is for two-way sync to a `@State` in a parent — not for ViewModel communication

---

## Side Effects

- `.task` modifier for async work triggered by view appearance — not `onAppear + Task { }`
- `.task(id:)` when the async work depends on a changing value
- `.onChange(of:)` for reacting to state changes with side effects
- Never start `Task { }` inside `body` — it runs on every re-render

---

## Avoiding Common Mistakes

- Never store server data in `@State` — use a ViewModel or model layer
- Never use `@EnvironmentObject` in new code — use `@Environment(MyType.self)` with `@Observable`
- Avoid `@StateObject` in new code unless targeting iOS < 17
- `@AppStorage` for simple user preferences only — not for structured app state
