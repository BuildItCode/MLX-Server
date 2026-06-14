# Deployment (expo/skills — expo-deployment)

Source: expo/skills (MIT) — https://github.com/expo/skills

---

## EAS Build

```json
// eas.json
{
  "cli": {
    "version": ">= 16.0.1",
    "appVersionSource": "remote"
  },
  "build": {
    "production": {
      "autoIncrement": true
    },
    "development": {
      "autoIncrement": true,
      "developmentClient": true
    }
  },
  "submit": {
    "production": {}
  }
}
```

```bash
# Build for production
eas build --platform all --profile production

# Build for development (custom dev client)
eas build --platform ios --profile development
eas build --platform android --profile development

# Build locally (requires Xcode / Android Studio)
eas build -p ios --profile development --local
eas build -p android --profile development --local
```

---

## When to Use EAS vs Expo Go

| Scenario | Use |
|---|---|
| All `expo-*` packages, Expo Router, Reanimated | Expo Go (`npx expo start`) |
| Custom native modules in `modules/` | EAS Build |
| Apple targets (widgets, app clips) | EAS Build |
| Third-party native modules not in Expo Go | EAS Build |
| Production submission | EAS Build |

**Always try Expo Go first** — EAS builds add complexity and require Xcode/Android Studio.

---

## EAS Update (OTA)

```bash
# Publish update to production channel
eas update --branch production --message "Fix: login crash"

# Publish to preview channel
eas update --branch preview --message "Add dark mode"
```

- OTA updates work for JS/assets only — native code changes require a new build
- `expo-updates` configured in `app.json` with `updates.url` pointing to EAS

---

## App Store Submission

```bash
# Submit to App Store (requires Apple credentials in eas.json or env)
eas submit -p ios --latest

# Submit to Google Play
eas submit -p android --latest
```

- `app.json` `expo.ios.bundleIdentifier` and `expo.android.package` must match store listings
- `autoIncrement: true` in `eas.json` manages version codes automatically
- App Store screenshots and metadata managed via EAS Metadata or manually

---

## Environment Variables in EAS

```bash
# Set secret in EAS (not stored in eas.json)
eas secret:create --scope project --name API_KEY --value "secret"

# List secrets
eas secret:list
```

- Secrets set via `eas secret:create` — never in `eas.json` or source
- `EXPO_PUBLIC_` vars in `.env` are bundled at build time — safe for non-secrets only

---

## CI/CD

```yaml
# Example GitHub Actions step
- name: Build and submit
  run: |
    npx eas-cli build --platform all --non-interactive
    npx eas-cli submit --platform all --non-interactive
  env:
    EXPO_TOKEN: ${{ secrets.EXPO_TOKEN }}
```

- `EXPO_TOKEN` from EAS dashboard — used for CI authentication
- `--non-interactive` flag for CI environments
