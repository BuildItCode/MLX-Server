# SDK Upgrades (expo/skills — upgrading-expo)

Source: expo/skills (MIT) — https://github.com/expo/skills

---

## Upgrade Command

```bash
# Upgrade to latest SDK
npx expo upgrade

# Fix dependency versions after upgrade
npx expo install --fix

# Check for incompatible versions
npx expo install --check
```

Always use `npx expo upgrade` — not manual `npm install expo@latest`.
`npx expo install --fix` resolves peer dependency mismatches automatically.

---

## New Architecture (SDK 55+ / RN 0.82+)

New Architecture is **mandatory** from SDK 55 / RN 0.82. There is no opt-out.

```bash
# This will FAIL in SDK 55+ / RN 0.82+:
# gradle.properties
newArchEnabled=false          # ❌ Ignored — build fails

# iOS
RCT_NEW_ARCH_ENABLED=0        # ❌ Ignored — build fails
```

If a third-party library does not support New Architecture, file an issue and find an alternative — downgrading is not a viable strategy.

---

## Breaking Changes Checklist (expo/skills)

### Deprecated Packages — Migrate Before Upgrading
- `expo-av` → `expo-audio` (audio) + `expo-video` (video)
  - `Audio.Sound` → `useAudioPlayer`
  - `Audio.Recording` → `useAudioRecorder`
  - `Video` component → `VideoView` with `useVideoPlayer`
- `expo-symbols` → `expo-image` with `source="sf:name"` for SF Symbols
- `@expo/vector-icons` → `expo-image` with SF Symbols (iOS) or Material Icons

### SDK 53+
- `resolver.unstable_enablePackageExports` enabled by default in Metro — remove from `metro.config.js`
- `cjs` and `mjs` extensions supported by default

### SDK 54+
- `EXPO_USE_FAST_RESOLVER=1` removed — delete from env
- `expo-av` deprecated — migrate to `expo-audio` / `expo-video`

### SDK 55+
- New Architecture mandatory
- `NativeTabs.Trigger.*` for `Icon`, `Label`, `Badge` (not top-level imports)
- Hermes v1 opt-in: `"jsEngine": "hermes"` in `app.json` for improved runtime performance

---

## Excluded Packages Check (expo/skills)

```json
// Check for excluded packages in package.json — often outdated workarounds
{
  "expo": {
    "install": {
      "exclude": ["react-native-reanimated"]  // Review after upgrade — may no longer be needed
    }
  }
}
```

Review every exclusion after upgrade — they are often workarounds that are no longer necessary.

---

## Patches Directory

```bash
# Check for outdated patches
ls patches/

# autoprefixer not needed in SDK 53+ — remove from dependencies
# postcss.config.js → postcss.config.mjs in SDK 53+
```

---

## CNG vs Bare Workflow

```bash
# Check if project uses CNG (Continuous Native Generation)
ls ios/    # exists? → Bare workflow
ls android/  # exists? → Bare workflow
# Neither exists? → CNG — native projects regenerated at build time
```

- **CNG** (no `ios/`/`android/`): Run `npx expo prebuild` to regenerate; `npx expo run:ios` also regenerates
- **Bare workflow**: Manually update `ios/` and `android/` using Expo upgrade guide

---

## Verifying the Upgrade

```bash
npx expo-doctor                    # checks for common issues
npx expo install --check           # checks dependency compatibility
npx tsc --noEmit                   # TypeScript check
npx expo start                     # test in Expo Go first
```
