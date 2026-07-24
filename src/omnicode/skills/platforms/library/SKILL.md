---
name: library
description: >
  Library / SDK / npm package code quality rules. Loaded automatically when platform=library.
  Covers public API design, backwards compatibility, tree-shaking, and documentation standards.
---

# Library / SDK Quality


## Public API Design

- Every exported symbol is intentional — unexported by default, exported by explicit decision
- Public API is minimal: solve one problem, solve it completely
- No internal implementation details in the public API surface
- Breaking changes require a major version bump (semver strictly followed)
- Deprecate before removing: `@deprecated` + migration path in the doc comment

## Backwards Compatibility

- New optional parameters always go at the end with a default
- New overloads rather than changing existing signatures
- `CHANGELOG.md` updated with every change; breaking changes marked clearly
- API surface tested with consumer-style tests (test the public interface, not internals)

## Tree-Shaking & Bundle

- Named exports only — no default export of a namespace object
- Side-effect-free modules declared with `"sideEffects": false` in `package.json`
- No top-level code that runs on import beyond constant definitions
- Peer dependencies declared correctly — not bundled

## Documentation

- `README.md` has: install, quickstart (working code), full API reference, and licence
- Every exported type, function, and class has a doc comment with at least one usage example
- TypeScript types shipped with the package (`types` field in `package.json`)
- Changelog follows Keep a Changelog format

## Validation Checks

```
✓ No internal types in public API surface
✓ All exports have doc comments with examples
✓ sideEffects: false in package.json
✓ Named exports only (no default namespace export)
✓ CHANGELOG.md updated
✓ Types field set in package.json
```
