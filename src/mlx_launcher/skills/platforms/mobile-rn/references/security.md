# Security

---

## Dependencies

- `npm audit` / `yarn audit` before every milestone — zero high/critical CVEs
- Exact version pinning in `package.json` — no `^` or `~` on production deps
- No packages unmaintained for > 2 years without an active successor
- `npx expo install --check` to verify Expo-compatible versions

---

## Sensitive Data

- Auth tokens → `expo-secure-store` (iOS Keychain / Android Keystore backed)
- `AsyncStorage` only for non-sensitive UI preferences (theme, locale, onboarding state)
- No secrets, API keys, or tokens in source code or committed `.env` files
- No PII in logs, analytics events, or crash reports — redact before sending
- `EXPO_PUBLIC_` prefix: only for client-safe, non-secret values

---

## Network

- All API calls over HTTPS — no HTTP in production
- Explicit request timeout on every `fetch` call — no indefinite waits
- Deep link parameters always validated before acting on them

---

## WebView (if used)

- `expo-web-browser` for external URLs — not embedded WebView
- If WebView required: set `originWhitelist` explicitly; `javaScriptEnabled` only when needed
- User-generated content never rendered as raw HTML

---

## Release Builds

- Expo Application Services (EAS) Build for production builds — not local `expo run`
- `eas.json` profiles separate `development`, `preview`, and `production`
- Source maps stored securely in EAS — never shipped in the JS bundle
- Enable Hermes for all production builds (default in SDK 50+)
- ProGuard / R8 for Android: enabled by default in EAS production profile
