---
name: ios
description: >
  iOS-specific code quality rules for Swift + SwiftUI projects. Loaded automatically
  when platform=mobile-ios. Extends universal quality rules — read quality/SKILL.md first.
  Incorporates guidance from twostraws/SwiftUI-Agent-Skill (MIT license). Targets iOS 18+,
  Swift 6, SwiftUI with @Observable. References load on demand — consult only the files
  relevant to the task at hand.
---

# iOS Quality — Swift + SwiftUI


## Review Workflow (twostraws/SwiftUI-Agent-Skill pattern)

When reviewing or writing iOS code, work through these in order. Load only the reference
files that apply to the current task — do not load all of them speculatively.

1. **Views & modifiers** → `references/views.md`
2. **Data flow & state** → `references/data.md`
3. **Navigation** → `references/navigation.md`
4. **Design & HIG** → `references/design.md`
5. **Accessibility** → `references/accessibility.md`
6. **Performance** → `references/performance.md`
7. **Swift language** → `references/swift.md`
8. **Code hygiene** → `references/hygiene.md`

Paths are relative to this file's location. In Claude Code: `.agents/skills/quality/platforms/ios/references/`.
In OpenCode: `.agents/skills/quality/platforms/ios/references/`.

Report only genuine problems. Do not nitpick or invent issues.

---

## Target Platform

- iOS 18.0+ / Swift 6.2+ / Xcode 16+
- `@Observable` macro for all shared data — not `ObservableObject`
- `async/await` for all async work — no `DispatchQueue` or closure-based APIs
- No UIKit unless explicitly required by the task
- No third-party frameworks without explicit approval in ARCH.md

---

## Validation Checklist

Run after every build task. Report only failures.

```
✓ No forced unwrap (!) in production code
✓ No DispatchQueue.main.async — @MainActor used throughout
✓ All @Observable classes marked @MainActor
✓ No ObservableObject in new code — @Observable used
✓ No foregroundColor() — foregroundStyle() used
✓ No Text concatenation with + — use string interpolation or Text("") + Text("")
✓ No GeometryReader where containerRelativeFrame() or visualEffect() suffices
✓ No Array(x.enumerated()) in ForEach — use ForEach(x.enumerated(), id: \.element.id)
✓ scrollIndicators(.hidden) not showsIndicators: false
✓ NavigationStack used — not NavigationView
✓ sheet(item:) with enum-backed binding — not sheet(isPresented:) with separate state
✓ List/LazyVStack not ScrollView + VStack for variable-length content
✓ All interactive elements have .accessibilityLabel
✓ Dynamic Type supported — no fixed font sizes
✓ .task modifier for async work in views — not onAppear + Task { }
✓ No business logic in View body
✓ Every View has #Preview
✓ Auth tokens in Keychain — not UserDefaults
✓ No NSAllowsArbitraryLoads in Info.plist
✓ No PII in os_log or analytics
```
