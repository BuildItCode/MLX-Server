# Performance

Source: twostraws/SwiftUI-Agent-Skill (MIT), Apple developer documentation

---

## @Observable Granularity

- `@Observable` only re-renders views that read a changed property — not the whole view tree
- Split large models into smaller focused ones to minimise unnecessary re-renders
- Never do expensive work inside `body` — precompute in the model, cache with `@State`
- Views with independent update cycles should be separate types, not inline closures

---

## View Initialiser Rule

- No work in `View.init` that touches `@State` or `@Observable` — it runs on every render cycle
- Expensive object creation goes in a model, not a view initialiser
- Never call `async` functions from `init` — use `.task` in `body`

---

## Lists & Lazy Layouts

- `List` recycles cells — prefer it over `ScrollView + LazyVStack` for uniform row content
- `LazyVStack` / `LazyHStack` only when content is non-uniform or requires custom container behaviour
- Stable, unique `id` for `ForEach` — identity changes cause full cell recreation, not update
- `@Observable` model in a list: each cell subscribes only to its item's properties — no whole-list re-render

---

## Images

- `AsyncImage` for remote images; always provide `content` and `placeholder` closures
- Downsample before displaying large images: create a downsampled `UIImage` sized to display bounds
- Never load images synchronously on the main thread
- `resizable().scaledToFit()` or `.scaledToFill()` — never unconstrained `Image`

---

## Instruments & Profiling

- Profile with **Instruments → SwiftUI** template to find recomposition storms
- `_logChanges()` in a view's `body` during debugging to identify what triggered a re-render (remove before committing)
- Hang Detector in Instruments catches main-thread blocking > 250ms

---

## Concurrency

- All async work off the main actor — `@MainActor` for UI updates only
- `Task.detached` only when explicitly not inheriting actor context — document why
- Long operations: `Task { await performHeavyWork() }` started from `.task` modifier
- Never use `DispatchQueue.global().async` — use Swift structured concurrency
