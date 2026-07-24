# Views & Modifiers

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple HIG

---

## Deprecated API — Always Replace

| Deprecated | Use instead |
|---|---|
| `.foregroundColor()` | `.foregroundStyle()` |
| `Text("a") + Text("b")` | String interpolation or `Text("a") + Text("b")` via `AttributedString` |
| `NavigationView` | `NavigationStack` or `NavigationSplitView` |
| `GeometryReader` (for relative sizing) | `containerRelativeFrame()` or `visualEffect()` |
| `showsIndicators: false` (ScrollView init) | `.scrollIndicators(.hidden)` |
| `.accentColor()` | `.tint()` |
| `UIGraphicsImageRenderer` | `ImageRenderer` |
| `Label("text", systemImage:)` with `.labelStyle(.iconOnly)` | `Image(systemName:)` directly when no text is needed |
| `.fontWeight(.bold)` | `.bold()` |

---

## View Composition

- Extract any `body` exceeding ~30 lines into a named private `View` or `@ViewBuilder` function
- `@ViewBuilder` functions in the same file for layout fragments closely tied to the parent
- Reusable components in separate files, grouped by feature — not by type
- Never define a View inside another View's `body` — always extract to a named type or `@ViewBuilder`

---

## Modifiers

- Custom `ViewModifier` for any modifier chain applied in more than one place
- `.frame(maxWidth: .infinity)` always paired with explicit `alignment` and a layout reason
- Never use `Spacer()` for fixed spacing — use `.padding()` or `.frame(height:)` on the parent

---

## Animations

- `.animation(.spring, value:)` for value-driven animations — not `.animation(.spring)` (deprecated)
- `withAnimation { }` for explicit state-change animations
- `matchedGeometryEffect` requires a `@Namespace` — never share a namespace across unrelated views
- Honor `@Environment(\.accessibilityReduceMotion)` — wrap non-essential animations in a check

---

## Images & SF Symbols

- `Image(systemName:)` for SF Symbols — always set `.accessibilityLabel` or `.accessibilityHidden(true)` for decorative use
- `AsyncImage` for remote images with `placeholder` and `failure` views — never load images synchronously
- Downsample large images before displaying: `Image(uiImage:)` with `UIImage(data:)` scaled to display size

---

## Lists & ForEach

- `List` or `LazyVStack` for variable-length content — not `ScrollView + VStack`
- `ForEach` requires stable `id` — `\.self` only for value types that are inherently unique
- **Never** convert to `Array` first: `ForEach(x.enumerated(), id: \.element.id)` not `ForEach(Array(x.enumerated()), ...)`
- `Table` (iOS 16+) for multi-column data on iPad/Mac Catalyst

---

## ScrollView

- `ScrollPosition` and `defaultScrollAnchor` for programmatic scrolling — not `ScrollViewReader`
- `.scrollIndicators(.hidden)` — not `showsIndicators: false` in the initialiser

---

## Buttons & Controls

- `Button` for all tappable elements — never `onTapGesture` on a non-interactive element
- `ButtonStyle` for reusable button appearance — never copy-paste modifier chains
- `confirmationDialog` for destructive actions — not `Alert` with destructive button only
