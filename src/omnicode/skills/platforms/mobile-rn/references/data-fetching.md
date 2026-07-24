# Data Fetching & Networking (expo/skills — native-data-fetching)

Source: expo/skills (MIT) — https://github.com/expo/skills

---

## Decision Tree (expo/skills)

```
Network request needed?
├── Route-level data loading (SDK 55+ web)?  → Expo Router loaders
├── Basic one-off fetch?                     → fetch API with error handling
├── Need caching / deduplication?
│   ├── Complex app                          → TanStack Query (React Query)
│   └── Simpler needs                        → SWR or custom hook
├── Authentication?
│   ├── Token storage                        → expo-secure-store
│   └── Token refresh                        → Refresh flow with singleton promise
├── Offline support?
│   ├── Check status                         → expo-network or NetInfo
│   └── Queue / pause requests               → React Query onlineManager
└── API config?
    ├── Client-safe URLs                     → EXPO_PUBLIC_ in .env
    └── Secrets                              → Non-prefixed (API routes only)
```

---

## Preferences (expo/skills)

- **`fetch` API** preferred over `axios` — use `expo/fetch` for Worker/Edge compatibility
- No `axios` unless team has an existing, well-tested wrapper

---

## Basic Fetch

```typescript
// ✓ typed, error-handling fetch
const fetchUser = async (userId: string): Promise<User> => {
  const response = await fetch(`${API_URL}/users/${userId}`);
  if (!response.ok) throw new ApiError(response.status, await response.text());
  return userSchema.parse(await response.json());  // validated, not cast
};

// ✓ POST
const createUser = async (data: CreateUserInput): Promise<User> => {
  const token = await SecureStore.getItemAsync('auth_token');
  const response = await fetch(`${API_URL}/users`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new ApiError(response.status, await response.text());
  return userSchema.parse(await response.json());
};
```

---

## TanStack Query (React Query) Setup

```tsx
// app/_layout.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { focusManager } from '@tanstack/react-query';
import { AppState } from 'react-native';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 2,
    },
  },
});

// Refetch on app foreground
AppState.addEventListener('change', status => {
  focusManager.setFocused(status === 'active');
});

export default function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <Stack />
    </QueryClientProvider>
  );
}

// Usage
function UserProfile({ userId }: { userId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
  });
  if (isLoading) return <Loading />;
  if (error) return <ErrorView message={error.message} />;
  return <Profile user={data!} />;
}
```

---

## Offline Support (expo/skills)

```typescript
import { onlineManager } from '@tanstack/react-query';
import * as Network from 'expo-network';

// Sync React Query with Expo network status
onlineManager.setEventListener(setOnline => {
  let initialised = false;
  const sub = Network.addNetworkStateListener(state => {
    initialised = true;
    setOnline(!!state.isConnected);
  });
  Network.getNetworkStateAsync()
    .then(state => { if (!initialised) setOnline(!!state.isConnected); })
    .catch(() => {});
  return sub.remove;
});
```

- Queries pause automatically when offline and resume on reconnect
- `useMutation` with `onError` + retry for queued write operations

---

## Authentication

```typescript
// Token storage
await SecureStore.setItemAsync('auth_token', token);
const token = await SecureStore.getItemAsync('auth_token');

// Token refresh — singleton promise prevents multiple parallel refreshes
let refreshPromise: Promise<string> | null = null;

const getValidToken = async (): Promise<string> => {
  const token = await SecureStore.getItemAsync('auth_token');
  if (token && !isTokenExpired(token)) return token;
  if (!refreshPromise) {
    refreshPromise = refreshToken().finally(() => { refreshPromise = null; });
  }
  return refreshPromise;
};
```

---

## Environment Config (expo/skills)

```bash
# .env
EXPO_PUBLIC_API_URL=https://api.example.com   # client-safe, inlined at build

# .env.development
EXPO_PUBLIC_API_URL=http://localhost:3000

# .env.production  
EXPO_PUBLIC_API_URL=https://api.production.com
```

```typescript
const API_URL = process.env.EXPO_PUBLIC_API_URL;
```

- `EXPO_PUBLIC_` prefix: safe for client bundle — never put secrets here
- Server secrets (API keys): non-prefixed, only accessible in Expo Router server functions/API routes
- Never hardcode API URLs in source — always environment variables

---

## Request Cancellation

```typescript
useEffect(() => {
  const controller = new AbortController();
  fetch(url, { signal: controller.signal })
    .then(r => r.json()).then(setData)
    .catch(err => { if (err.name !== 'AbortError') setError(err); });
  return () => controller.abort();
}, [url]);
```
