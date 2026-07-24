# Swift Language

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple developer documentation

---

## Concurrency

- `async/await` for all asynchronous work — no `DispatchQueue` or completion-handler callbacks in new code
- `@MainActor` on any class or function touching UI state
- `actor` for shared mutable state accessed from multiple concurrent contexts
- `@Sendable` on closures that cross actor boundaries
- `async let` for independent parallel work — not nested `Task` chains
- `TaskGroup` when number of concurrent operations is dynamic
- Never `DispatchQueue.main.async` — use `await MainActor.run { }` or `@MainActor` annotation

---

## Optionals

- Forced unwrap (`!`) never in production — use `guard let`, `if let`, `??`, `map`, or `flatMap`
- `guard let` at function entry for early exit; `if let` for mid-function branching
- `try!` never in production; `try?` only when `nil` on failure is the correct semantic
- `@discardableResult` only when callers genuinely have no use for the return value

---

## Types

- `struct` by default; `class` only when reference semantics or inheritance are required
- `enum` with associated values for exhaustive state modelling
- Protocol extensions over subclassing
- `final` on classes where inheritance is not intended
- `typealias` for complex generic types used across files

---

## Modern APIs (iOS 18+ / Swift 6)

- `String(localized:)` — not `NSLocalizedString`
- `#Preview` macro — not `PreviewProvider` (unless backward compatibility required)
- Swift Testing (`@Test`, `@Suite`) — not `XCTestCase` for new test files
- `SwiftData` — not Core Data for new persistence in new projects
- Typed throws (`throws(AppError)`) where the error type is known

---

## Code Organisation

- One type per file — no multiple structs, classes, or enums in a single file
- Extensions grouped by protocol conformance in `// MARK: - ProtocolName` sections
- `defer` for cleanup that must always run regardless of exit path
- `@discardableResult`, `@autoclosure`, `@escaping` only when the semantic is correct

---

## Naming

- Types and protocols: `UpperCamelCase`
- Properties and functions: `lowerCamelCase`
- Boolean properties: `isVisible`, `hasNotifications`, `canSubmit`
- Never abbreviate unless the abbreviation is universally known (`URL`, `ID`, `API`)
- SwiftData model properties: always provide default values or mark optional — never `@Attribute(.unique)`
