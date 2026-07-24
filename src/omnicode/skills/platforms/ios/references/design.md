# Design & Human Interface Guidelines

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple HIG

---

## HIG Compliance Rules

- Interactive touch targets: minimum 44×44pt — use `.contentShape(Rectangle())` to expand hit area without changing appearance
- All text in body style or above supports Dynamic Type — never use `.font(.system(size: 14))` with fixed size
- Dark mode supported — all colours use semantic colour assets or `Color(.label)`, `Color(.systemBackground)` etc.
- Never hardcode `Color(red:green:blue:)` for UI elements — use semantic colours or custom colour assets with dark/light variants
- Avoid colour as the sole indicator of meaning — pair with text or shape

---

## Typography

- Use `.font(Font.TextStyle)` — not fixed sizes: `.font(.body)`, `.font(.headline)`, `.font(.caption)` etc
- Custom fonts must declare `UIFontMetrics` scaling: `Font.custom("MyFont", size: 16, relativeTo: .body)`
- Bold via `.bold()` not `.fontWeight(.bold)` — (twostraws rule: more idiomatic SwiftUI)

---

## Spacing & Layout

- 8pt grid — spacing values should be multiples of 4 or 8 (`4, 8, 12, 16, 20, 24, 32, 40, 48`)
- Use `ViewThatFits` for adaptive layouts that need to respond to size class changes
- `containerRelativeFrame()` for proportional sizing — not `GeometryReader`

---

## Colours & Materials

- System materials (`.ultraThinMaterial`, `.regularMaterial`) for frosted-glass effects
- `Color.primary`, `Color.secondary` for text — not custom colours for body text
- Test against both light and dark themes, and high-contrast mode

---

## Icons & Symbols

- SF Symbols preferred — consistent with system visual language
- Match SF Symbol weight to surrounding font weight: `.imageScale(.medium)` with `.font(.body)`
- Never use a filled symbol where an outlined one is expected by convention (e.g. tab bar: outlined when unselected, filled when selected)

---

## Design Principles (arjitj2/swiftui-design-principles)

- **Clarity**: UI communicates purpose without decoration — remove any element that doesn't serve the user
- **Deference**: UI recedes so content is primary — avoid heavy chrome, prefer subtle backgrounds and transitions
- **Depth**: Layers and motion communicate hierarchy — use sheets, split views, and navigation to show relationships, not flat modals stacked arbitrarily
- **Consistency**: Use system components before custom ones — `List`, `Form`, `Picker`, `Toggle` behave as users expect
- **Feedback**: Every interaction has immediate visual response — use `.buttonStyle(.bordered)` press states, `.sensoryFeedback`, `withAnimation` on state changes
- **Forgiveness**: Destructive actions require confirmation — `confirmationDialog` before delete/reset

---

## Localisation

- All user-facing strings via `String(localized:)` or `Text(.helloWorld)` using string catalog symbol keys with `extractionState: manual`
- Never `NSLocalizedString` in new SwiftUI code
- RTL support: use leading/trailing instead of left/right; test with a right-to-left locale
- Plurals declared in the string catalog — not manual `count == 1 ? "item" : "items"`
