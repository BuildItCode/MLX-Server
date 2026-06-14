# TypeScript

---

## Required tsconfig

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  }
}
```

Note: `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` are intentionally
omitted — they produce excessive noise with RN's generated types and community libraries.
Enable only after evaluating the trade-off for your project.

---

## Type Discipline

- `any` never — production code, tests, or mocks
- `unknown` + narrowing for values without a known compile-time type
- `as` assertions only at validated API boundaries (after `zod.parse`) — never to silence errors
- `!` non-null assertions never — use `?.`, `??`, or explicit guards
- `@ts-ignore` requires a comment explaining why; prefer `@ts-expect-error`

```typescript
// ✓ API response validated at boundary
const userSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
});
type User = z.infer<typeof userSchema>;

// ✓ typed catch
catch (error: unknown) {
  if (error instanceof ApiError) handleApiError(error);
  else throw new UnexpectedError(String(error));
}

// ✗ unsafe — bypasses validation
const user = response.data as User;
const name = user!.name;
```

---

## Component & Hook Typing

- Component props: `interface {ComponentName}Props` — not inline object types
- Hook return values typed explicitly when non-trivial
- `Record<K, V>` over `{ [key: string]: V }`
- `Readonly<T>` on state shapes that must not be mutated
- `satisfies` for config objects that should be validated without widening

---

## Environment Variables

```typescript
// types/env.d.ts — extend ProcessEnv for autocomplete
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      EXPO_PUBLIC_API_URL: string;
      EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY: string;
    }
  }
}
```

- `EXPO_PUBLIC_` prefix for client-safe values only — inlined at build time
- Never put secrets in `EXPO_PUBLIC_` vars — they are visible in the JS bundle
- Server secrets in non-prefixed vars — only accessible in API routes (Expo Router server functions)
