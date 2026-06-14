# Building Native UI (expo/skills — building-native-ui)

Source: expo/skills (MIT) — https://github.com/expo/skills

---

## Library Preferences (expo/skills)

| Use | Instead of |
|---|---|
| `expo-image` | `<img>`, `react-native` Image (for SF Symbols use `source="sf:name"`) |
| `expo-audio` | `expo-av` Audio |
| `expo-video` | `expo-av` Video |
| `react-native-safe-area-context` | RN `SafeAreaView` |
| `process.env.EXPO_OS` | `Platform.OS` |
| `React.use` | `React.useContext` |
| `expo-glass-effect` | Custom blur implementations |

**Never use modules removed from React Native**: `Picker`, `WebView` (RN's), `SafeAreaView` (RN's), `AsyncStorage` (RN's).
**Never use `expo-permissions`** (legacy) — use individual permission packages.

---

## Expo Go First (expo/skills rule)

```
CRITICAL: Always try Expo Go before creating custom builds.

npx expo start → scan QR with Expo Go → test thoroughly

Custom builds (npx expo run:ios / eas build) ONLY when using:
- Local Expo modules (custom native code in modules/)
- Apple targets (@bacons/apple-targets)
- Third-party native modules not in Expo Go
- Custom native configuration not expressible in app.json
```

---

## Responsiveness (expo/skills)

- Always wrap root component in a `ScrollView` for responsiveness
- `<ScrollView contentInsetAdjustmentBehavior="automatic" />` instead of `<SafeAreaView>`
- Apply `contentInsetAdjustmentBehavior="automatic"` to `FlatList` and `SectionList` too
- `useWindowDimensions` over `Dimensions.get()` — reactive to size changes
- `flexbox` over `Dimensions` API for layout

```tsx
// ✓ first child of a Stack route
export default function HomeScreen() {
  return (
    <ScrollView contentInsetAdjustmentBehavior="automatic">
      <ThemedView>
        {/* content */}
      </ThemedView>
    </ScrollView>
  );
}
```

---

## Styling (expo/skills)

```tsx
// ✓ CSS boxShadow — the only correct shadow API
<View style={{ boxShadow: "0 1px 2px rgba(0, 0, 0, 0.05)" }} />

// ✗ legacy — removed/deprecated
<View style={{ shadowColor: '#000', shadowOffset: ..., elevation: 2 }} />
```

- CSS `boxShadow` style prop — never legacy `shadow*` or `elevation` styles
- `{ borderCurve: 'continuous' }` for rounded corners (iOS squircle)
- Inline styles for one-off dynamic values; `StyleSheet.create` for static reusable styles
- CSS and Tailwind not supported in React Native — use inline styles or NativeWind
- Counters: `{ fontVariant: 'tabular-nums' }` for numeric alignment
- Padding a `ScrollView`: use `contentContainerStyle` padding — not padding on the ScrollView itself

---

## Navigation (expo/skills)

```tsx
// ✓ Link for navigation
import { Link } from 'expo-router';
<Link href="/settings" />

// ✓ Link with context menu and preview (iOS conventions)
<Link href="/item/123" asChild>
  <Link.Trigger>
    <Pressable><ItemCard /></Pressable>
  </Link.Trigger>
  <Link.Preview />
  <Link.Menu>
    <Link.MenuAction title="Share" icon="square.and.arrow.up" onPress={handleShare} />
    <Link.MenuAction title="Delete" icon="trash" destructive onPress={handleDelete} />
  </Link.Menu>
</Link>
```

- `<Link href="/path" />` from `expo-router` — not `router.push`
- Include `<Link.Preview>` wherever possible — follows iOS conventions
- Stack title via `<Stack.Screen options={{ title: "..." }} />` — not custom Text on page
- `<Stack.Screen options={{ presentation: "formSheet" }}>` for sheets — not custom modal components

---

## Native Tabs (expo/skills — SDK 55+)

```tsx
// app/_layout.tsx
import { NativeTabs, Icon, Label } from "expo-router/unstable-native-tabs";

export default function Layout() {
  return (
    <NativeTabs>
      <NativeTabs.Trigger name="(index)">
        <Icon sf="list.dash" />
        <Label>Items</Label>
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="(search)" role="search" />
    </NativeTabs>
  );
}
```

---

## Haptics (expo/skills)

- Use `expo-haptics` conditionally on iOS for delightful interactions
- Built-in haptics from `<Switch />` (RN) and `@react-native-community/datetimepicker`
- `Haptics.impactAsync(ImpactFeedbackStyle.Light)` on significant actions

---

## Text

- `<Text selectable />` on text containing data users might want to copy
- Format large numbers: `1.4M`, `38k` — not raw `1400000`
- Never use intrinsic `img` or `div` elements — except in Expo DOM components (`"use dom"`)

---

## Code Style (expo/skills)

- kebab-case for all file names: `comment-card.tsx`
- Import statements at top of file always
- Path aliases in `tsconfig.json` — prefer aliases over relative imports for refactors
- Never co-locate components, types, or utilities in the `app/` directory — anti-pattern
- Remove old route files when restructuring navigation
