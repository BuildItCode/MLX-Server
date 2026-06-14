# Jetpack Compose

---

## Composable Design

- One visual responsibility per `@Composable` — if it needs "and", split it
- Composables are stateless by default; state lives in the ViewModel or is hoisted to the caller
- No business logic inside a composable — presentation logic only (formatting, conditional rendering)
- No direct ViewModel access in leaf composables — pass state and lambdas down

## State Hoisting

```kotlin
// ✓ stateless leaf — state and events flow through parameters
@Composable
fun EmailField(
    value: String,
    onValueChange: (String) -> Unit,
    isError: Boolean = false,
    modifier: Modifier = Modifier,
)

// ✗ composable owns state it doesn't need to own
@Composable
fun EmailField() {
    var value by remember { mutableStateOf("") }
}
```

## Recomposition

- `remember` for values surviving recomposition; `rememberSaveable` for process death
- `derivedStateOf { }` for computations depending on other state — not plain `remember`
- Lambdas passed to children use `rememberUpdatedState` or are stable references
- `key(id) { }` in `LazyColumn`/`LazyRow` when items have identity
- Never read `State` inside a lambda that outlives the composition scope

## Modifiers

- `modifier: Modifier = Modifier` always the **last** non-trailing-lambda parameter
- No hardcoded `Modifier.size()` where `fillMaxWidth`/`wrapContentSize` is correct
- `Modifier.semantics { }` on every interactive element lacking native semantics

## Theming

- All colours, typography, spacing via `MaterialTheme` — no hardcoded hex values or `dp` literals
- Custom tokens via `CompositionLocalProvider` — not global `object` constants
- Every composable with visual appearance has `@Preview(uiMode = UI_MODE_NIGHT_YES)`

## Previews

- Every leaf composable has at least one `@Preview`
- `@PreviewParameter` covering empty, loading, error, and data states
- Previews use `PreviewData` objects or fake factories — never real ViewModels or repositories

## Performance

- No allocations inside composable bodies that run on every recomposition — use `remember { }`
- `LazyColumn`/`LazyRow` for lists that may exceed ~10 items
- Images via Coil `AsyncImage` with `contentScale`, `placeholder`, `error`
- Animations via `Modifier.graphicsLayer` or `Animatable` — never animate layout properties directly
- Baseline Profiles for critical user journeys
