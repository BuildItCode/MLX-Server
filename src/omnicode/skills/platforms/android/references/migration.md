# Migration Guides

Covers: XML Views to Compose, Play Billing upgrade

---

## XML Views to Jetpack Compose (android/skills)

Migrate incrementally using `ComposeView` / `AndroidView` interop. Never big-bang.

```kotlin
// ✓ Embed Compose in a Fragment during migration
class ProfileFragment : Fragment() {
    override fun onCreateView(...) = ComposeView(requireContext()).apply {
        setContent { AppTheme { ProfileScreen(viewModel = viewModel()) } }
    }
}

// ✓ Embed a legacy View inside Compose
@Composable
fun LegacyMapView(modifier: Modifier = Modifier) {
    AndroidView(
        factory = { context -> MapView(context).apply { onCreate(null) } },
        modifier = modifier,
        update = { view -> view.getMapAsync { map -> /* configure */ } },
    )
}
```

**Widget mapping:**

| XML | Compose |
|---|---|
| `TextView` | `Text` |
| `ImageView` | `Image` / `AsyncImage` (Coil) |
| `Button` | `Button` / `FilledButton` |
| `RecyclerView` | `LazyColumn` / `LazyRow` |
| `ConstraintLayout` | `ConstraintLayout` (compose) |
| `LinearLayout` (vertical) | `Column` |
| `LinearLayout` (horizontal) | `Row` |
| `FrameLayout` | `Box` |
| `ScrollView` | `Column` + `verticalScroll` or `LazyColumn` |
| `LiveData.observe` | `collectAsStateWithLifecycle()` |
| `Fragment` | Screen `@Composable` |

**Rules:**
- One screen migrated at a time
- Fragment hosting `ComposeView` should have no XML layout other than the `ComposeView`
- Migrate UI tests from Espresso to Compose test APIs per screen
- Remove `viewBinding`/`dataBinding` from a module only after all screens are migrated

---

## Play Billing Library 7+ (android/skills)

```toml
[versions]
billing = "7.1.1"
[libraries]
billing = { module = "com.android.billingclient:billing-ktx", version.ref = "billing" }
```

```kotlin
class BillingRepository(context: Context) {
    private val billingClient = BillingClient.newBuilder(context)
        .setListener(::onPurchasesUpdated)
        .enablePendingPurchases(
            PendingPurchasesParams.newBuilder().enableOneTimeProducts().build()
        )
        .build()

    private fun onPurchasesUpdated(result: BillingResult, purchases: List<Purchase>?) {
        if (result.responseCode == BillingResponseCode.OK && purchases != null) {
            purchases.forEach { processPurchase(it) }
        }
    }
}
```

**Rules:**
- Always verify purchases server-side before granting entitlements
- Acknowledge purchases within 3 days — Google refunds unacknowledged purchases
- `PENDING` state must be handled — grant entitlement only when `PURCHASED`

**Migration from v5/v6:**

| Old | New |
|---|---|
| `enablePendingPurchases()` (no-arg) | `enablePendingPurchases(PendingPurchasesParams...)` |
| `SkuDetails` | `ProductDetails` |
| `queryProductDetailsAsync` | same, with `ProductDetailsParams` |
| `queryPurchases` (sync) | `queryPurchasesAsync` |
| `Purchase.purchaseState` | `Purchase.getPurchaseState()` |
