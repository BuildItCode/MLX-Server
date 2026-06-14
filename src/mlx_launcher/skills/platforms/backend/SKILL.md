---
name: backend
description: >
  Backend code quality rules for Node.js/TypeScript, Python, and Go projects.
  Loaded automatically when platform=backend. Extends universal quality rules —
  read quality/SKILL.md first. Language-specific type rules, security, API design,
  error handling, database, observability, and framework patterns.
---

# Backend Quality — Server-Side


## Type Safety

### TypeScript / Node.js

Required `tsconfig.json`:
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noImplicitReturns": true,
    "noUncheckedIndexedAccess": true,
    "forceConsistentCasingInFileNames": true,
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16"
  }
}
```

- `any` never — use `unknown` + narrowing for external data
- `zod` for runtime validation at all trust boundaries: request bodies, env vars, API responses
- `@ts-ignore` requires a comment; prefer `@ts-expect-error`

```typescript
// ✓ env vars validated at startup — crash-fast on misconfiguration
const envSchema = z.object({
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  PORT: z.coerce.number().int().min(1024).max(65535).default(3000),
});
export const env = envSchema.parse(process.env);

// ✓ request body parsed, not cast
const createUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(['admin', 'member']),
});
// type CreateUserInput = z.infer<typeof createUserSchema>
```

### Python

- `mypy --strict` passes with zero errors in CI
- All function parameters and return types annotated
- `from __future__ import annotations` at the top of every file
- `pydantic` v2 for all data validation — `BaseModel` for DTOs and config
- `Literal`, `TypeVar`, `Generic`, `Protocol` for reusable typed abstractions
- `Any` from `typing` never in production — use `object` or proper generics

```python
# ✓ typed, validated
class CreateUserInput(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    role: Literal['admin', 'member']

# ✓ typed return, typed error
async def get_user(user_id: UUID) -> User:
    user = await db.users.find_by_id(user_id)
    if user is None:
        raise NotFoundError(entity='User', id=str(user_id))
    return user
```

### Go

- No `interface{}` or `any` in public API or function signatures — concrete types or typed interfaces
- All errors returned — never ignored; `_` on an error return requires a comment
- `fmt.Errorf("context: %w", err)` for error wrapping — `errors.Is` / `errors.As` for inspection
- `context.Context` as the first parameter of every function performing I/O
- Structured types for all request/response shapes — no `map[string]interface{}`

```go
// ✓ typed, context-aware
func (r *UserRepository) GetByID(ctx context.Context, id uuid.UUID) (*User, error) {
    var user User
    err := r.db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id,
    ).Scan(&user.ID, &user.Name, &user.Email)
    if errors.Is(err, sql.ErrNoRows) {
        return nil, ErrNotFound
    }
    return &user, err
}
```

---

## Security (All Languages)

### Input Validation
- All external input validated before reaching business logic — at the handler / controller boundary
- Unknown fields in request bodies rejected (`z.object().strict()` / `extra='forbid'` / struct tags)
- File uploads: MIME type validated by content sniffing (not extension), size limited, stored outside webroot

### SQL Injection Prevention
```typescript
// ✓ parameterised
await db.query('SELECT * FROM users WHERE email = $1', [email]);

// ✗ critical violation
await db.query(`SELECT * FROM users WHERE email = '${email}'`);
```
- Parameterised queries always — string interpolation into SQL is a critical contract violation
- Raw SQL only when the ORM cannot express the query; must still use parameterised form

### Authentication & Authorisation
- Auth validated in middleware — never per-handler inline checks
- JWT: `exp`, `iss`, `aud` all validated; secrets rotated via env var, not hardcoded
- Passwords hashed with `bcrypt` (cost ≥ 12) or `argon2id` — MD5, SHA1, SHA256 never for passwords
- Authorisation (RBAC / ABAC) enforced in the service layer — controllers handle HTTP concerns only
- Rate limiting on: login, registration, password reset, email verification, and all public endpoints

### Secrets
- Secrets from environment variables or a secrets manager (Vault, AWS SSM, GCP Secret Manager)
- Never hardcoded, never in source files, never in committed `.env` files
- `.env.example` with dummy values committed; real `.env` in `.gitignore`
- Env vars validated at startup — the process crashes on missing required secrets

### HTTP Security Headers (set on every response)
```
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'; object-src 'none'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### Dependency Security
- `npm audit` / `pip audit` / `govulncheck` before every milestone — zero high/critical CVEs
- Exact version pinning in production

---

## API Design

### REST Conventions
- Resources are nouns, HTTP verbs are actions: `GET /users/:id`, `POST /users`, `DELETE /users/:id`
- Consistent response envelope for all endpoints:
```typescript
type ApiSuccess<T> = { data: T; meta?: PaginationMeta };
type ApiError    = { error: { code: string; message: string; details?: unknown } };
```
- Pagination on all list endpoints — cursor-based preferred over offset for large datasets
- Versioning via URL prefix (`/v1/`) — breaking changes require a new version, never silent changes
- `4xx` for client errors with actionable, non-technical messages; `5xx` with no internal details leaked

### Input / Output Contracts
- Request schemas defined with zod / pydantic / Go struct tags and used to drive documentation
- Response types inferred from schemas — no manual synchronisation between types and runtime
- Nullable and optional distinguished explicitly — `null` means absent-but-known; `undefined`/omitted means not applicable

### GraphQL (when applicable)
- Schema-first: `.graphql` schema file is the contract; resolvers implement it
- `DataLoader` for all relationships to eliminate N+1 — no per-resolver fetches
- Depth and complexity limits configured on the server
- `graphql-codegen` for typed resolvers and typed client operations

---

## Error Handling

### Error Hierarchy (TypeScript example — apply equivalent in Python/Go)
```typescript
// Domain errors — no HTTP knowledge
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: unknown,
  ) { super(message); this.name = this.constructor.name; }
}
class NotFoundError  extends AppError { constructor(msg: string) { super(msg, 'NOT_FOUND'); } }
class ConflictError  extends AppError { constructor(msg: string) { super(msg, 'CONFLICT'); } }
class ForbiddenError extends AppError { constructor(msg: string) { super(msg, 'FORBIDDEN'); } }

// HTTP mapping only at the controller boundary
const HTTP_STATUS: Record<string, number> = {
  NOT_FOUND: 404, CONFLICT: 409, FORBIDDEN: 403,
};
function toHttpResponse(err: AppError) {
  return {
    status: HTTP_STATUS[err.code] ?? 500,
    body: { error: { code: err.code, message: err.message } },
  };
}
```

- Business logic throws domain errors — never `HttpException` or framework HTTP errors inside services
- Global error handler catches, maps, and responds — no leaked stack traces in any environment
- Unhandled promise rejections / panics crash the process — let the process manager restart cleanly

---

## Database

- Migrations versioned, reversible, and committed to source control (`prisma migrate` / `alembic` / `golang-migrate`)
- No raw queries in business logic — repository pattern with a typed interface in domain
- N+1 queries always resolved: `include` / `select`, DataLoader, or explicit JOIN
- Transactions for any operation that requires multiple writes to remain consistent
- Connection pool size configured and tuned — not left at the ORM default
- Query timeouts set on the connection/client — no unbounded query execution
- Columns containing PII or credentials excluded from default `SELECT *` — use explicit projections

---

## Observability

### Structured Logging
```typescript
// ✓ structured JSON — pino (Node.js) / structlog (Python) / slog (Go)
logger.info('User created', { userId: user.id, requestId: ctx.requestId });

// ✗ unstructured, no context
console.log('User created: ' + user.id);
```

- Every log line includes: `level`, `message`, `timestamp`, `service`, `requestId`
- Log levels used correctly: `debug` (dev only), `info` (key lifecycle events), `warn` (recoverable degradation), `error` (requires attention)
- No PII, passwords, or tokens in any log line

### Metrics & Tracing
- Request count, error rate, and latency (p50/p95/p99) instrumented via Prometheus or OpenTelemetry
- Distributed tracing with W3C `traceparent` header propagated through all service calls
- `GET /health` returns `{ status: 'ok', checks: { db: 'ok', cache: 'ok' } }` — used by load balancers
- `GET /metrics` (internal network only) exposes Prometheus metrics

---

## Framework Patterns

### Node.js — Fastify (preferred over Express for new projects)
- Route handlers stay thin: validate input → call service → format response
- Plugins use `fastify-plugin` for proper encapsulation and dependency injection
- Schema validation declared on every route with a JSON Schema or zod adapter — not manually inside handlers
- Error handling via a single `setErrorHandler` registered on the root instance

### Python — FastAPI
- `APIRouter` per feature module — no monolithic `main.py` with all routes
- `Depends()` for all dependency injection: database sessions, auth, services
- `response_model` declared on every endpoint — controls serialisation, excludes private fields
- Background tasks: `BackgroundTasks` for fire-and-forget; `celery` or `arq` for reliable async work

### Go — net/http + Chi / Gin
- Handler signature `func(w http.ResponseWriter, r *http.Request)` — no framework lock-in in domain code
- Middleware via standard `http.Handler` wrapping — not framework decorators
- `context.Context` propagated through all layers for cancellation and deadline
- Sentinel errors (`var ErrNotFound = errors.New("not found")`) for domain errors — compared with `errors.Is`
- Goroutines always have a clear owner and shutdown path — never fire-and-forget without a `WaitGroup` or context

---

## Testing

| Layer | Tool (TS / Python / Go) | Scope |
|---|---|---|
| Unit | Jest / pytest / `testing` | Services, use cases, validators, mappers |
| Integration | Supertest / httpx / `httptest` | HTTP handlers with real or test DB |
| Contract | Pact | Service boundaries (if microservices) |
| Load | k6 | Critical endpoints before release |

- Integration tests run against a real database in a Docker container — not mocks
- Test database is isolated per test run; migrations applied fresh
- No `process.env.NODE_ENV === 'test'` branches in production code

---

## Validation Checks

```
✓ TypeScript: strict tsconfig; no any; env vars validated at startup with zod
✓ Python: mypy --strict passes; pydantic v2 with extra='forbid'; no Any
✓ Go: no interface{}/any in public APIs; all errors returned and checked
✓ All external input validated at handler boundary before business logic
✓ No string interpolation in SQL queries — parameterised always
✓ npm/pip/go audit: zero high/critical CVEs
✓ No secrets hardcoded or in committed files; validated at startup
✓ All HTTP security headers present on every response
✓ Auth in middleware — not per-handler; JWT exp/iss/aud validated
✓ Passwords: bcrypt cost ≥ 12 or argon2id — no weaker algorithms
✓ Rate limiting on auth and public endpoints
✓ Structured JSON logging — no console.log in production
✓ requestId on every log line; no PII in logs
✓ N+1 queries resolved; transactions for multi-step mutations
✓ Query timeout set on DB client; connection pool configured
✓ Global error handler; no stack traces in any response body
✓ GET /health with dependency checks
✓ Prometheus metrics or OpenTelemetry instrumented
```
