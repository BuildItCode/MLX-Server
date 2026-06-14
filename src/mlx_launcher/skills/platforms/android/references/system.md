# System Integration

Covers: Edge-to-Edge, AGP 9, R8, Security, Platform Rules

---

## Edge-to-Edge (android/skills — mandatory Android 16+)

`windowOptOutEdgeToEdgeEnforcement` is deprecated on Android 16 (API 36) — no opt-out.

```kotlin
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge() // must be before setContent
        setContent {
            AppTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    AppContent(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}
```

```kotlin
// ✓ WindowInsets for system bar padding
NavigationBar(
    modifier = Modifier.windowInsetsPadding(WindowInsets.navigationBars)
)

// ✓ FlatList / LazyColumn below system bars
LazyColumn(contentPadding = WindowInsets.systemBars.asPaddingValues()) { ... }
```

**Inset types:**

| Type | Use |
|---|---|
| `.systemBars` | Status + navigation bar combined |
| `.navigationBars` | Bottom bar / gesture area |
| `.statusBars` | Top status bar |
| `.ime` | Keyboard (for scrollable content) |
| `.safeDrawing` | All areas including cutout |

**Rules:** `enableEdgeToEdge()` before `setContent`. No `fitsSystemWindows="true"` on root with edge-to-edge. Remove `windowOptOutEdgeToEdgeEnforcement = true` if present.

---

## AGP 9 Upgrade (android/skills)

**Requirements:** AGP ≥ 9.0, Gradle ≥ 9.1, JDK 17, KSP ≥ 2.3.6

**Key changes:**

```kotlin
// ✗ Remove — kotlin-android is now built into AGP 9
plugins { id("org.jetbrains.kotlin.android") version "2.x.x" }

// ✗ Remove — old variant API removed
android.applicationVariants.all { variant -> ... }

// ✓ Use androidComponents API instead
androidComponents {
    onVariants { variant ->
        variant.buildConfigFields.put("FIELD", BuildConfigField("String", "\"value\"", null))
    }
}
```

**Migration steps:**
1. Use AGP Upgrade Assistant in Android Studio
2. Bump `agp`, `gradle`, `kotlin`, `ksp` in `libs.versions.toml`
3. Remove `kotlin-android` plugin from all modules
4. Replace `applicationVariants`/`libraryVariants` with `androidComponents.onVariants {}`
5. KMP modules: migrate to `com.android.kotlin.multiplatform.library`
6. Remove legacy flags from `gradle.properties`:
   - `android.enableLegacyVariantApi=true`
   - `android.builtInKotlin=false`
   - `android.newDsl=false`

**R8 in AGP 9.1:** Classes repackaged into default package by default. Opt out: add `-dontrepackage` to ProGuard rules.

---

## R8 / ProGuard (android/skills)

```kotlin
// build.gradle.kts (app module)
android {
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
}
```

```proguard
# AGP 9+ — remove Kotlin null checks in release
-processkotlinnullchecks remove
```

- Never `-keep class ** { *; }` — defeats shrinking
- `@Keep` annotation over manual rules where possible
- Debug symbols: `mappingFileOutput = file("mapping.txt")` — store securely, never ship
- De-obfuscate crash stacks: `retrace mapping.txt stacktrace.txt`

---

## Security

- Sensitive data: `EncryptedSharedPreferences` (AES256_SIV) or Android Keystore
- Network: HTTPS always; `CertificatePinner` for APIs handling sensitive data
- No PII, tokens, or secrets in logs — Timber `DebugTree` in debug builds only
- `android:debuggable="false"` enforced in release manifest
- Dependencies: audit before each milestone — zero high/critical CVEs

---

## Platform Rules

- No logic in `Activity` or `Fragment` beyond navigation setup and entry-point wiring
- `onBackPressedDispatcher` for back handling — never override `onBackPressed()`
- Permissions via `rememberLauncherForActivityResult`; result handled in ViewModel
- `WorkManager` for background work — never `Service` for one-shot tasks
