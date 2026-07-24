# Accessibility (flutter/skills)

Source: flutter/skills (Apache 2.0), flutter.dev/accessibility

---

## Screen Readers (flutter/skills — flutter-accessibility skill)

- Every interactive widget has `Semantics` label or inherits one from a parent `Semantics` wrapper
- `Semantics(label: 'Delete item', child: IconButton(...))` — not relying on icon name alone
- Decorative images: `ExcludeSemantics(child: Image(...))` or `Semantics(image: true, label: '', child: ...)`
- Custom controls: `Semantics(button: true, onTap: ..., label: ...)` for full TalkBack/VoiceOver support
- `MergeSemantics` to combine adjacent semantics nodes into one readable unit

```dart
// ✓ button with explicit label
Semantics(
  label: 'Add to favourites',
  button: true,
  child: IconButton(icon: const Icon(Icons.favorite_border), onPressed: onFavourite),
)

// ✓ decorative image
ExcludeSemantics(child: Image.asset('assets/background.png'))

// ✓ merged row — reads as one unit
MergeSemantics(
  child: Row(children: [
    const Icon(Icons.check),
    Text(item.name),
  ]),
)
```

---

## Dynamic Text Scaling

- All text uses `Theme.of(context).textTheme` styles — no hardcoded `fontSize`
- Layouts must not overflow at `textScaleFactor` up to 2.0 — test with Accessibility > Large Text
- `MediaQuery.textScalerOf(context)` to react to scale changes if needed
- Scrollable containers for long text — never fixed-height boxes that clip

---

## Colour & Contrast

- Minimum contrast ratio 4.5:1 for normal text, 3:1 for large text (WCAG AA)
- Colour never the sole indicator of meaning — pair with icon, label, or pattern
- Test with `MediaQuery.highContrast` and Invert Colours on device

---

## Touch Targets

- Minimum 48×48dp tappable area — use `InkWell` with `SizedBox(width: 48, height: 48)` or padding
- `MaterialButton` and `IconButton` meet 48dp by default — do not shrink with `constraints: BoxConstraints()`
- Custom tap targets: wrap with `GestureDetector` + `SizedBox(width: 48, height: 48)`

---

## Focus & Keyboard

- `FocusNode` for programmatic focus management in forms
- Tab order follows visual order — use `FocusTraversalGroup` to fix non-sequential order
- All interactive elements reachable by keyboard on desktop and web targets

---

## Testing Accessibility

```bash
# Run semantic tree validation
flutter test --tags=accessibility

# Enable in widget tests
tester.binding.defaultBinaryMessenger;
final SemanticsHandle handle = tester.ensureSemantics();
// ... test ...
handle.dispose();
```

- `tester.ensureSemantics()` in widget tests to validate semantic tree
- Test with TalkBack (Android) and VoiceOver (iOS) on physical devices before task complete
- Check touch target sizes with Flutter Inspector → select widget → check render box
