---
name: mobile-rn
description: >
  React Native + Expo code quality rules. Loaded automatically when platform=mobile-rn.
  Extends universal quality rules — read quality/SKILL.md first. Incorporates official
  guidance from expo/skills (MIT). Covers TypeScript, UI, data fetching, navigation,
  security, deployment, and SDK upgrades. Reference files load on demand.
---

# React Native Quality — TypeScript + Expo + React Native


## Review Workflow (expo/skills pattern)

Load only the reference files relevant to the current task.

1. **TypeScript & types** → `references/typescript.md`
2. **Building UI** → `references/ui.md` (expo/skills building-native-ui)
3. **Data fetching & networking** → `references/data-fetching.md` (expo/skills native-data-fetching)
4. **Navigation** → `references/navigation.md` (expo-router)
5. **Security** → `references/security.md`
6. **Deployment** → `references/deployment.md` (expo/skills expo-deployment)
7. **SDK upgrades** → `references/upgrading.md` (expo/skills upgrading-expo)

Paths relative to this file. In Claude Code: `.agents/skills/quality/platforms/mobile-rn/references/`.
In OpenCode: `.agents/skills/quality/platforms/mobile-rn/references/`.

---

## Target Platform

- Expo SDK 55+ / React Native 0.79+ / React 19+
- New Architecture mandatory (SDK 55+ / RN 0.82+) — no opt-out
- TypeScript strict mode always
- Expo Router for file-based navigation
- `expo-image` over `<img>`, `expo-audio` over `expo-av`, `expo-video` over `expo-av`
- `process.env.EXPO_OS` over `Platform.OS`
- Try Expo Go first — custom builds only when native modules require it

---

## Validation Checklist

```
✓ tsconfig: strict, noImplicitAny, noImplicitReturns, noFallthroughCasesInSwitch
✓ No any, @ts-ignore (without comment), or non-null assertions (!) in production
✓ npm/yarn audit: zero high/critical CVEs; exact version pinning
✓ No Picker, WebView (RN), SafeAreaView (RN), AsyncStorage (RN) — removed from RN
✓ expo-image not <img>; expo-audio not expo-av; expo-video not expo-av
✓ contentInsetAdjustmentBehavior="automatic" on ScrollView/FlatList/SectionList
✓ No legacy React Native shadow/elevation — use CSS boxShadow style prop
✓ No class components; no component defined inside another component
✓ useEffect deps exhaustive; cleanup returned for subscriptions and timers
✓ FlashList/FlatList for scrollable lists — no ScrollView + map
✓ StyleSheet.create for static styles; no inline objects; no hex literals in components
✓ All routes in app/ directory — no components/types/utils co-located in app/
✓ Route entry: first child is ScrollView with contentInsetAdjustmentBehavior="automatic"
✓ Auth tokens in expo-secure-store — not AsyncStorage
✓ All API calls over HTTPS; EXPO_PUBLIC_ prefix for client-safe env vars only
✓ Secrets never in source or committed .env files
✓ eas.json configured; EAS Build used for production
✓ npx expo upgrade or npx expo install --fix for dependency upgrades
```
