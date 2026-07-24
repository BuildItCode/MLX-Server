# Accessibility

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple developer documentation

---

## VoiceOver

- Every interactive element has `.accessibilityLabel` — descriptive, not generic ("Delete", not "Button")
- `.accessibilityHint` for non-obvious actions — not needed if the label already explains it
- Decorative images: `.accessibilityHidden(true)` so VoiceOver skips them
- Grouped content: `.accessibilityElement(children: .combine)` to prevent VoiceOver reading each child separately
- Custom controls: `.accessibilityRepresentation { }` for full VoiceOver compatibility
- Never make VoiceOver-invisible interactive elements — test that every tap target is reachable
- `.accessibilityAction(named:)` for custom swipe actions

---

## Dynamic Type

- All text uses `Font.TextStyle` — never `.font(.system(size: 14))`
- Custom fonts scaled with `UIFontMetrics`: `Font.custom("Name", size: 16, relativeTo: .body)`
- Images and icons that sit next to text scale with `.imageScale()` or use `Layout.scalesWithDynamicType`
- Verify layout does not break at `Accessibility Extra Extra Extra Large` size

---

## Reduce Motion

```swift
@Environment(\.accessibilityReduceMotion) var reduceMotion

var body: some View {
    content
        .animation(reduceMotion ? .none : .spring, value: isExpanded)
}
```

- All non-essential animations must respect `accessibilityReduceMotion`
- Auto-playing videos and animated backgrounds must pause when reduce motion is enabled

---

## Contrast & Colour

- Minimum contrast ratio: 4.5:1 for normal text, 3:1 for large text (WCAG AA)
- Never use colour as the only differentiator — pair with shape, label, or pattern
- Test with `Increase Contrast` accessibility setting enabled

---

## Touch Targets

- Minimum 44×44pt tappable area — use `.contentShape(Rectangle())` to expand without changing appearance
- Never shrink interactive controls below 44pt on any axis
- `TapGesture` on small elements needs explicit `.frame(minWidth: 44, minHeight: 44)`

---

## Focus & Keyboard

- `@FocusState` for programmatic focus management in forms
- Logical focus order — `.accessibilitySortPriority()` to fix non-sequential reading order
- Hardware keyboard: `.keyboardShortcut()` on primary actions in Mac Catalyst and iPad

---

## Testing

- Test every screen with VoiceOver on a physical device before task complete
- Use Xcode Accessibility Inspector to audit labels and hit areas
- `XCUIAccessibilityElement` in UI tests to verify element existence by label
