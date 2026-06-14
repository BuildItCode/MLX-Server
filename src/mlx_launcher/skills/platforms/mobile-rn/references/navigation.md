# Navigation — Expo Router (expo/skills)

Source: expo/skills (MIT) — https://github.com/expo/skills

---

## File-Based Routing

```
app/
  _layout.tsx              Root layout — NativeTabs or Stack
  index.tsx                Matches "/"
  (auth)/
    _layout.tsx            Auth group layout
    login.tsx              Matches "/login"
  (tabs)/
    _layout.tsx            Tab layout
    (index,search)/
      _layout.tsx          Shared stack for index and search tabs
      index.tsx            Tab index screen
      search.tsx           Search screen
      i/[id].tsx           Detail screen (dynamic route)
  +not-found.tsx           404 handler
```

**Rules (expo/skills):**
- Routes belong in the `app/` directory only
- Never co-locate components, types, or utilities in `app/` — anti-pattern
- Every app must have a route matching `/` (may be inside a group)
- Remove old route files when restructuring — stale routes cause routing bugs

---

## Stack Navigation

```tsx
// app/(tabs)/(index,search)/_layout.tsx
import { Stack } from 'expo-router/stack';
import { PlatformColor } from 'react-native';

export default function Layout({ segment }) {
  const screen = segment.match(/\((.*)\)/)?.[1]!;
  const titles: Record<string, string> = { index: 'Items', search: 'Search' };

  return (
    <Stack
      screenOptions={{
        headerTransparent: true,
        headerShadowVisible: false,
        headerLargeTitleShadowVisible: false,
        headerLargeStyle: { backgroundColor: 'transparent' },
        headerTitleStyle: { color: PlatformColor('label') },
        headerLargeTitle: true,
        headerBlurEffect: 'none',
        headerBackButtonDisplayMode: 'minimal',
      }}
    >
      <Stack.Screen name={screen} options={{ title: titles[screen] }} />
      <Stack.Screen name="i/[id]" options={{ headerLargeTitle: false }} />
    </Stack>
  );
}
```

- Always use `_layout.tsx` files to define stacks — never inline stack definitions
- Set page title in `Stack.Screen options` — not a custom Text element on the page
- `headerSearchBarOptions` for search bars — not custom search implementations

---

## Modals and Sheets (expo/skills)

```tsx
// Modal
<Stack.Screen name="modal" options={{ presentation: 'modal' }} />

// Form sheet with detents
<Stack.Screen
  name="sheet"
  options={{
    presentation: 'formSheet',
    sheetGrabberVisible: true,
    sheetAllowedDetents: [0.5, 1.0],
    contentStyle: { backgroundColor: 'transparent' },  // liquid glass on iOS 26+
  }}
/>
```

- Prefer `presentation: 'formSheet'` over custom modal components
- `contentStyle: { backgroundColor: 'transparent' }` enables liquid glass backdrop on iOS 26+

---

## Programmatic Navigation

```tsx
import { router } from 'expo-router';

// Push
router.push('/settings');
router.push({ pathname: '/users/[id]', params: { id: '123' } });

// Replace (no back navigation)
router.replace('/login');

// Back
router.back();
router.canGoBack() ? router.back() : router.replace('/');
```

---

## Link with Context Menu (expo/skills)

```tsx
import { Link } from 'expo-router';

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

Include `<Link.Preview>` frequently — it follows iOS navigation conventions.

---

## Dynamic Routes

```
app/users/[id].tsx           → /users/123
app/users/[...rest].tsx      → /users/a/b/c (catch-all)
app/(group)/index.tsx        → / (group doesn't appear in URL)
```

---

## Deep Links

```json
// app.json
{
  "expo": {
    "scheme": "myapp",
    "web": { "bundler": "metro" }
  }
}
```

Deep link params validated before acting on them — never trust raw URL parameters.
