# Layouts & Widgets (flutter/skills)

Source: flutter/skills (Apache 2.0)

---

## The Constraint Rule

**Constraints go down. Sizes go up. Parent sets position.**

- Parent passes constraints (min/max width+height) to children
- Widget reports desired size back within those constraints
- Widget cannot choose its own position — parent sets it
- **Never pass unbounded constraints** in the cross-axis of a flex (Row/Column) or inside scrollable regions

---

## Stateless First

```dart
// ✓ StatelessWidget with const constructor — default choice
class UserCard extends StatelessWidget {
  const UserCard({required this.user, required this.onTap, super.key});
  final User user;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) { ... }
}
```

Use `StatefulWidget` only for:
- `AnimationController` / `TickerProvider`
- `TextEditingController`, `FocusNode`, `ScrollController`
- `initState` / `dispose` for resource lifecycle
- `AutomaticKeepAliveClientMixin`

---

## Widget Extraction

```dart
// ✓ named private widget — extracted when build() > 40 lines
class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title});
  final String title;

  @override
  Widget build(BuildContext context) => Text(title, style: Theme.of(context).textTheme.titleMedium);
}

// ✗ anonymous Builder for non-trivial content
child: Builder(builder: (ctx) => Column(children: [/* 30 lines */])),
```

---

## const Everywhere

```dart
// ✓ const on all eligible instantiations
const Text('Hello', style: TextStyle(fontSize: 16)),
const SizedBox(height: 8),
const Icon(Icons.check, color: Colors.green),

// ✗ unnecessary allocations every build
Text('Hello', style: TextStyle(fontSize: 16)),
SizedBox(height: 8),
```

---

## Lists

- `ListView.builder` / `GridView.builder` for dynamic lists — `ListView(children: list.map(...))` never
- `SliverList` with `SliverChildBuilderDelegate` for custom scroll effects
- `CustomScrollView` with slivers for mixed content (app bar + list + grid)

---

## Constraint Gotchas

```dart
// ✗ Column inside Column with unbounded height — throws RenderFlex overflow
Column(children: [
  Column(children: [...]),  // inner Column has unbounded height
])

// ✓ wrap inner list in Expanded or a fixed-height container
Column(children: [
  Expanded(child: ListView.builder(...)),
])

// ✗ ListView inside Column without Expanded/fixed size
Column(children: [
  ListView(children: [...]),  // unbounded height — throws
])

// ✓
Column(children: [
  Expanded(child: ListView(children: [...])),
])
```

---

## Theming

- All colours from `Theme.of(context).colorScheme` — no hardcoded `Color(0xFF...)` in widgets
- All text styles from `Theme.of(context).textTheme` — no hardcoded `fontSize` literals
- Spacing from a `Spacing` constants class — multiples of 4 or 8
- `borderRadius: BorderRadius.circular(AppRadius.md)` — no magic numbers

---

## Previews

Every reusable widget has a preview:

```dart
// user_card.preview.dart
void main() => runApp(MaterialApp(
  theme: AppTheme.light,
  home: Scaffold(body: UserCard(user: User.fixture(), onTap: () {})),
));
```

- Covers: default, empty/null data, long content, dark theme
- Uses `.fixture()` factory methods — no real services
