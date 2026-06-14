# Code Hygiene

Source: twostraws/SwiftUI-Agent-Skill (MIT)

---

## Compilation

- Zero warnings — treat warnings as errors in CI (`SWIFT_TREAT_WARNINGS_AS_ERRORS = YES`)
- No `@MainActor` annotation missing on `@Observable` classes — compiler warning in Swift 6
- No `#if DEBUG` branching in production logic paths — only in test utilities and preview helpers

---

## Previews

- Every `View` has at least one `#Preview` macro block
- Cover at minimum: default state, empty/loading, error state, dark mode
- Previews use static fixture data or `PreviewProvider` helpers — no real network, no real persistence
- `#Preview(traits: .sizeThatFitsLayout)` for components that shouldn't be full-screen
- `#Preview("Dark Mode", traits: .colorScheme(.dark))` for dark mode variant

---

## File Structure

- One type per file — filename matches the type name
- Feature-based folders — not type-based (`Features/Login/` not `ViewModels/`)
- Secrets and API keys never in source — use `.xcconfig` or environment injection

---

## Strings

- User-facing strings via `String(localized:)` or string catalog symbol keys
- `Text(.localizedKey)` not `Text("hardcoded string")` for any text shown to users
- Never compute user-facing strings with `+` concatenation — use `Text` composition or `AttributedString`

---

## Imports

- Only import what is needed — no `import UIKit` in SwiftUI-only files
- No `@testable import` in production targets

---

## Comments & Documentation

- DocC comments (`///`) on all public types, functions, and properties
- No inline `// comments` that describe what the code does — refactor until the code is clear
- `// MARK: -` for logical section breaks within a file
- `// TODO:` only with a ticket reference — no orphaned TODOs

---

## Architecture Boundaries

- `View` types import only SwiftUI and their own domain types — no networking or persistence imports
- `@Observable` models import only Foundation and domain types — no SwiftUI import
- Repository protocol in domain layer; concrete implementation in data layer
- No `Singleton.shared` in ViewModels or domain — inject via `init` or `@Environment`
