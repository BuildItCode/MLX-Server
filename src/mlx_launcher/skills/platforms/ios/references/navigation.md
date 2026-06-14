# Navigation

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple developer documentation

---

## Stack Navigation

```swift
// ✓ NavigationStack with typed path
NavigationStack(path: $navigationPath) {
    ContentView()
        .navigationDestination(for: UserRoute.self) { route in
            UserDetailView(userId: route.userId)
        }
}

// Route type — Hashable value type
enum UserRoute: Hashable {
    case detail(userId: String)
    case settings
}
```

- `NavigationStack` for all new code — `NavigationView` is deprecated
- Typed `navigationDestination(for:)` — not string-based routing
- Programmatic navigation via `NavigationPath` or a typed `[Route]` array in the ViewModel

---

## Split View Navigation

- `NavigationSplitView` for iPad/Mac where sidebar + detail layout is appropriate
- `columnVisibility` to control initial sidebar state
- `navigationSplitViewStyle(.balanced)` vs `.prominentDetail` — choose based on content hierarchy

---

## Sheets & Overlays

```swift
// ✓ enum-backed sheet — single source of truth
enum SheetRoute: Identifiable {
    case addItem
    case editItem(Item)
    var id: String { ... }
}

@State private var sheet: SheetRoute?

.sheet(item: $sheet) { route in
    switch route {
    case .addItem: AddItemView()
    case .editItem(let item): EditItemView(item: item)
    }
}

// ✗ separate Bool + optional state — avoid
@State private var showAddItem = false
@State private var editingItem: Item?
```

- `sheet(item:)` with an `Identifiable` enum — not `sheet(isPresented:)` with separate state
- `fullScreenCover(item:)` same pattern
- `confirmationDialog` for destructive choices — not custom alert implementations
- `popover` on iPad where pointer interactions are expected

---

## Deep Links & State Restoration

- Decode deep link URLs into typed route values at the `App` level — not inside individual views
- `NavigationPath` supports `Codable` for state restoration across process death
- Test deep links with `UIApplication.shared.open()` in simulator

---

## Common Mistakes to Avoid

- Never conditionally wrap in `NavigationStack` — causes loss of navigation state
- Never use `NavigationLink(destination:isActive:)` (deprecated) — use `navigationDestination(for:)`
- Never push a `NavigationStack` inside another `NavigationStack` — use a single root stack with nested destinations
- Never pass `NavigationPath` or router down through many view layers — use `@Environment` injection
