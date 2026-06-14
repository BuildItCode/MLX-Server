---
name: web
description: >
  Web frontend code quality rules for React, Next.js, and Vue 3 projects.
  Loaded automatically when platform=web. Extends universal quality rules —
  read quality/SKILL.md first. Covers TypeScript configuration, security, component
  design, state management, accessibility, performance, and framework-specific patterns.
---

# Web Quality — React / Next.js / Vue 3


## TypeScript

### Required tsconfig
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  }
}
```

Note: `exactOptionalPropertyTypes` is beneficial but breaks many community types. Enable only
after auditing the project's dependency types. `noPropertyAccessFromIndexSignature` causes
friction with common DOM patterns — omit unless specifically required.

### Type Discipline
- `any` never — not in production code, tests, or mocks
- `unknown` + narrowing for values from external sources (API responses, events, env vars)
- `@ts-ignore` requires a comment; prefer `@ts-expect-error` (fails when the error is fixed)
- `as` type assertions only at validated API boundaries (after `zod.parse`) — never to silence errors
- `!` non-null assertions never — use optional chaining, `??`, or explicit guards

```typescript
// ✓ runtime validation at every trust boundary
const userSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
});
type User = z.infer<typeof userSchema>;

async function fetchUser(id: string): Promise<User> {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return userSchema.parse(await res.json());
}

// ✓ unknown in catch
catch (error: unknown) {
  if (error instanceof ApiError) toast.error(error.message);
  else throw error;
}
```

### Utility Types
- `Pick`, `Omit`, `Partial`, `Required`, `Readonly` over manual re-declaration
- `Record<K, V>` over index signatures for typed maps
- `satisfies` for config objects to validate against a type without widening
- `as const` for literal inference on tuples and config objects

---

## Security

### Dependencies
- `npm audit` / `pnpm audit` before every milestone — zero high/critical CVEs
- Exact versions in `package.json` — no `^` or `~` on production dependencies
- Supply-chain monitoring via Dependabot or Socket.dev

### XSS Prevention
- `dangerouslySetInnerHTML` never without first sanitising with `DOMPurify`
- No HTML strings constructed from user input injected into the DOM
- Content Security Policy header on all responses: `script-src 'self'; object-src 'none'`

### Authentication
- Auth tokens in `httpOnly; Secure; SameSite=Strict` cookies — never `localStorage`
- CSRF protection on all state-mutating endpoints
- OAuth / OIDC flows use PKCE — no implicit grant
- Session expiry handled gracefully: redirect to login, preserve post-login intent

### Environment Secrets
- Backend secrets never in the frontend bundle
- Only `NEXT_PUBLIC_` / `VITE_` prefixed variables (non-secret) are safe to bundle
- API keys that must be client-side → proxy through a backend endpoint
- `.env` files with real secrets never committed — `.env.example` with dummy values committed

### Third-Party Scripts
- All third-party `<script>` tags include `integrity` (SRI hash) and `crossorigin="anonymous"`
- No `eval()`, `new Function()`, or `import()` with user-controlled strings
- `postMessage` listeners always validate `event.origin` before processing

---

## React — Component Design

### Rules
- Functional components only — no class components
- One component per file; filename = PascalCase export name
- Props typed with `interface {ComponentName}Props` — not inline object types
- Props destructured in signature; defaults via destructuring — not `defaultProps`
- No component definitions inside another component's body (breaks memoisation)

### Server vs Client Components (Next.js App Router)
```typescript
// Server Component — default, no directive needed
// Can be async, fetches data directly, no useState/useEffect
async function UserProfile({ userId }: { userId: string }) {
  const user = await userRepository.getById(userId); // direct call
  return <UserCard user={user} />;
}

// Client Component — opt-in when browser APIs, state, or interactivity needed
'use client';
function LikeButton({ postId }: { postId: string }) {
  const [liked, setLiked] = useState(false);
  return <button onClick={() => setLiked(l => !l)}>{liked ? '❤️' : '🤍'}</button>;
}
```

- `'use client'` boundary pushed as deep in the component tree as possible
- Data fetching in Server Components — not `useEffect` + fetch in Client Components
- `Suspense` boundaries around async Server Components with meaningful `fallback` content

### Component Size (tighter than universal)
- Component body exceeding 80 lines → extract sub-components
- More than one concern (layout + data + interaction) → split into separate components

---

## Hooks (React)

- Custom hooks for all logic shared across 2+ components; named `use{Concept}`
- `eslint-plugin-react-hooks` enforced — `exhaustive-deps` never suppressed with a comment
- `useEffect` cleanup always returned for subscriptions, timers, and event listeners

```typescript
// ✓ abort controller for fetch — cancels on unmount and dep change
useEffect(() => {
  const controller = new AbortController();
  fetchUser(id, { signal: controller.signal })
    .then(data => setUser(userSchema.parse(data)))
    .catch(err => { if (err.name !== 'AbortError') setError(err); });
  return () => controller.abort();
}, [id]);
```

- `useMemo` / `useCallback` only when React DevTools profiler confirms a render cost
- `useRef` for mutable values that must not trigger re-renders

---

## State Management (React)

| Scope | Tool | Rule |
|---|---|---|
| Local UI | `useState` / `useReducer` | Never lift beyond where it's needed |
| Server state | TanStack Query | `queryKey` from factory functions — no raw inline arrays |
| Global | Zustand | One typed slice per domain |
| Cross-cutting | React Context | Theme, locale, auth session — not for frequent updates |

- Derived state computed in selectors or `useMemo` — never stored separately
- `useOptimistic` (React 19+) for optimistic mutations — not manual twin-state patterns

---

## Vue 3 (when framework = Vue 3)

```typescript
// ✓ Composition API with typed props and emits
<script setup lang="ts">
const props = defineProps<{
  user: User;
  isLoading?: boolean;
}>();

const emit = defineEmits<{
  select: [userId: string];
  dismiss: [];
}>();

const fullName = computed<string>(
  () => `${props.user.firstName} ${props.user.lastName}`
);
</script>
```

- `<script setup lang="ts">` on every component — no Options API for new code
- `defineProps<T>()` with generic type argument — no runtime `PropType` validators as sole typing
- `defineEmits<T>()` with generic type — every emitted event explicitly typed
- `composables/use{Concept}.ts` for all shared logic — no mixins
- Pinia for global state: `defineStore` with explicit state, getters, and actions types
- Vue Router 4 with typed routes via `RouteRecordRaw` and typed `useRoute` / `useRouter` calls

---

## Accessibility (Non-Negotiable)

- Interactive elements are keyboard navigable — no `div`/`span` with `onClick` without `role` + `tabIndex`
- All images: `alt` describing content, or `alt="" aria-hidden="true"` if decorative
- Form inputs have an associated `<label>` (via `for` / `htmlFor`) — `placeholder` never the sole label
- WCAG AA contrast: 4.5:1 for normal text, 3:1 for large text and UI components
- `aria-*` only when native HTML semantics are insufficient — no redundant `role="button"` on `<button>`
- Focus management on modal open/close: focus trapped inside, restored to trigger on close
- `prefers-reduced-motion` respected for all CSS and JS animations
- Test with keyboard-only navigation before marking any UI task complete

---

## Performance

- Route-level code splitting via `React.lazy` + `Suspense` or Next.js file-based splitting
- Images: `next/image` or explicit `width` / `height` attributes; `loading="lazy"` for below-fold
- Fonts: `font-display: swap`; subset with `unicode-range` where possible
- No synchronous `localStorage` / `sessionStorage` reads in render — use initialiser functions
- Core Web Vitals targets (enforced via Lighthouse CI): LCP < 2.5s, INP < 200ms, CLS < 0.1

### Next.js Cache
```typescript
// ✓ explicit cache control on every fetch
const user = await fetch(`/api/users/${id}`, {
  next: { revalidate: 60 },  // ISR — revalidate every 60s
});

const config = await fetch('/api/config', {
  cache: 'force-cache',  // never revalidate — static
});

const live = await fetch('/api/feed', {
  cache: 'no-store',  // dynamic — never cache
});
```
Every `fetch` in a Server Component has an explicit `cache` or `next.revalidate` option.

---

## CSS / Styling

- No magic numbers — spacing and type scale from design tokens (`--spacing-md`, `theme.spacing.md`)
- No `!important` — specificity problems solved by structure, not overrides
- No inline `style` for static values — CSS class or extracted CSS-in-JS
- Media queries use a consistent breakpoint scale defined once — no one-off values
- CSS custom properties for theme values — enables dark mode without JS class toggling
- CSS Modules or scoped styles — no bare global class names that risk collision

---

## Validation Checks

```
✓ tsconfig: strict, noImplicitAny, noUncheckedIndexedAccess, noImplicitReturns
✓ No any, @ts-ignore (without comment), or non-null assertions (!) in production
✓ npm/pnpm audit: zero high/critical CVEs
✓ No dangerouslySetInnerHTML without DOMPurify sanitisation
✓ Auth tokens in httpOnly cookies — not localStorage
✓ No secrets in frontend bundles (only NEXT_PUBLIC_ / VITE_ prefixed, non-secret vars)
✓ No class components (React); no Options API (Vue 3)
✓ No component defined inside another component's body
✓ useEffect deps exhaustive; cleanup returned for subscriptions and timers
✓ 'use client' pushed as deep as possible (Next.js App Router)
✓ Every fetch in Server Components has explicit cache or next.revalidate (Next.js)
✓ All images have alt text; all interactive elements keyboard navigable
✓ Form inputs have associated labels — not placeholder-only
✓ LCP < 2.5s, INP < 200ms, CLS < 0.1 (Lighthouse CI)
✓ No inline styles for static values; no magic numbers or hex literals in components
✓ Vue: defineProps<T> and defineEmits<T> with generic types; script setup lang="ts"
```
