---
name: mobile-flutter
description: >
  Flutter + Dart code quality rules. Loaded automatically when platform=mobile-flutter.
  Extends universal quality rules — read quality/SKILL.md first. Incorporates official
  guidance from flutter/skills (Apache 2.0). Covers architecture, state, navigation,
  layouts, testing, accessibility, and performance. Reference files load on demand.
---

# Flutter Quality — Dart + Flutter


## Review Workflow (flutter/skills pattern)

Load only the reference files relevant to the current task.

1. **Architecture & layers** → `references/architecture.md`
2. **State management** → `references/state.md`
3. **Navigation & routing** → `references/navigation.md`
4. **Layouts & widgets** → `references/layouts.md`
5. **Accessibility** → `references/accessibility.md`
6. **Performance** → `references/performance.md`
7. **Testing** → `references/testing.md`

Paths relative to this file. In Claude Code: `.agents/skills/quality/platforms/mobile-flutter/references/`.
In OpenCode: `.agents/skills/quality/platforms/mobile-flutter/references/`.

---

## Target Platform

- Flutter stable channel, Dart 3+, sound null safety
- `dart analyze --fatal-warnings` passes with zero issues in CI
- Architecture: MVVM + layered (UI / Domain / Data) per flutter/skills
- State: Riverpod (new projects) or BLoC/Cubit — declared in ARCH.md
- Navigation: GoRouter

---

## Validation Checklist

```
✓ dart analyze --fatal-warnings: zero errors, zero warnings
✓ No ! null assertions in production code
✓ No dynamic type in production code
✓ Every StatelessWidget has a const constructor
✓ const used on all eligible widget instantiations
✓ StatefulWidget only for local lifecycle (animation/text/scroll controllers)
✓ No business logic in build() methods
✓ build() methods ≤ 40 lines — named private widgets extracted beyond that
✓ Every reusable widget has a preview (.preview.dart or Widgetbook)
✓ Previews use .fixture() data — no real services
✓ Screens: ConsumerWidget / BlocProvider — not StatefulWidget for route state
✓ All Riverpod providers at file top level (not inside widgets or functions)
✓ Cubits/Blocs have no Flutter imports
✓ ViewModels (ChangeNotifier): no direct service calls — only via Repository
✓ Repositories: single source of truth, return Result types
✓ Services: stateless, no business logic
✓ GoRouter with go_router_builder typed routes — no raw context.go() strings
✓ flutter_secure_storage for credentials — not SharedPreferences
✓ HTTPS only; explicit timeouts on all Dio/http clients
✓ Release builds: --obfuscate --split-debug-info always
✓ ListView.builder for all dynamic lists
✓ compute() for heavy Dart work off the main isolate
✓ Fakes over mocks in tests — FakeRepository not MockRepository.when()
```
