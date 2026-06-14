# Navigation 3 (Official android/skills)

Source: https://github.com/android/skills — navigation/navigation-3

---

## Core Concepts

- **Back stack**: developer-owned `mutableStateListOf<NavKey>` — not framework-owned
- **NavKey**: `@Serializable` data class/object implementing `NavKey` interface
- **NavDisplay**: observes back stack and renders the top entry (or multiple for adaptive layouts)
- **NavEntry**: maps a key to composable content
- **Scene**: layout strategy (single pane, two pane, list-detail, dialog, bottom sheet)

## Navigation Keys

```kotlin
@Serializable data object Home : NavKey
@Serializable data class ProductDetail(val productId: String) : NavKey
@Serializable data class UserProfile(val userId: String, val showEdit: Boolean = false) : NavKey
```

## Back Stack

```kotlin
// Persistent across config changes and process death
val backStack = rememberNavBackStack(Home)

backStack.add(ProductDetail("abc"))    // navigate forward
backStack.removeLastOrNull()           // navigate back
```

## NavDisplay Setup

```kotlin
@Composable
fun AppNavigation() {
    val backStack = rememberNavBackStack(Home)
    NavDisplay(
        backStack = backStack,
        onBack = { backStack.removeLastOrNull() },
        entryProvider = entryProvider {
            entry<Home> { HomeScreen(onNavigate = { backStack.add(ProductDetail(it)) }) }
            entry<ProductDetail> { key -> ProductDetailScreen(productId = key.productId) }
        },
        entryDecorators = listOf(
            rememberSceneSetupNavEntryDecorator(),
            rememberSavedStateNavEntryDecorator(),
            rememberViewModelStoreNavEntryDecorator(),
        ),
    )
}
```

## Rules

- `rememberNavBackStack` always — not bare `remember { mutableStateListOf() }` (no process-death persistence)
- Always include the three standard decorators: SceneSetup, SavedState, ViewModelStore
- Check `backStack.size > 1` before `removeLastOrNull()` to prevent popping the last item
- Never modify `backStack` off the main thread
- Deep links: parse URL into a `NavKey` at the `Activity` entry point; pass the key, not the URL
- Auth/conditional navigation: redirect in `entryProvider` — not inside individual screens

## Migration from Navigation 2

| Nav 2 | Nav 3 |
|---|---|
| `navController.navigate(route)` | `backStack.add(NavKey)` |
| `navController.popBackStack()` | `backStack.removeLastOrNull()` |
| `NavHost { composable() }` | `NavDisplay { entryProvider { entry<>() } }` |
| String routes | `@Serializable` data classes implementing `NavKey` |
| Single pane only | Multi-pane via `SceneStrategy` |

## Dependencies

```toml
[versions]
navigation3 = "1.0.0"
lifecycleViewmodelNavigation3 = "1.0.0"
kotlinxSerializationJson = "1.8.1"

[libraries]
navigation3-runtime = { module = "androidx.navigation3:navigation3-runtime", version.ref = "navigation3" }
navigation3-ui = { module = "androidx.navigation3:navigation3-ui", version.ref = "navigation3" }
lifecycle-viewmodel-navigation3 = { module = "androidx.lifecycle:lifecycle-viewmodel-navigation3", version.ref = "lifecycleViewmodelNavigation3" }
kotlinx-serialization-json = { module = "org.jetbrains.kotlinx:kotlinx-serialization-json", version.ref = "kotlinxSerializationJson" }

[plugins]
kotlinSerialization = { id = "org.jetbrains.kotlin.plugin.serialization", version.ref = "kotlin" }
```
