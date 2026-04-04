# Feature Engineering Plan
## Higgsfield Churn Prediction — Complete Feature Specification

**Labels:** `not_churned` / `vol_churn` / `invol_churn`  
**Grain:** one row per `user_id`  
**Signal tags:** `VOL` = voluntary churn signal, `INV` = involuntary churn signal, `RET` = retention signal, `BOTH` = separates churned from not-churned regardless of type

> **Schema note:** `user_id` was added to `train_users_transaction_attempts.csv` and `test_users_transaction_attempts.csv` (now 20 columns). The EDA summary `.md` is stale and still shows 19 columns for the Transactions table — it does not yet reflect this addition. This changes how all transaction features are joined: previously required transactions → purchases → users; now transactions join directly to users. Features marked *(join changed)* below benefit from this.

---

## Platform Context (from research doc)

| Plan | Monthly Cost | Monthly Credits | Concurrent Jobs |
|---|---|---|---|
| Free | $0 | 300 (daily refresh) | 1 |
| Basic | ~$9 | 1,000 | 2 |
| Pro | ~$29 | 3,000 | 4 |
| Ultimate | ~$49 | 7,000 | 6 |
| Creator | ~$149 | 20,000 | 10+ |

**Approx credit costs per generation:**
- Standard image: 1–5 credits
- High-res image (4K): 10–15 credits
- Basic video (5s): 20–30 credits
- High-fidelity video (Veo/Sora): 50–100 credits

These values make exact plan utilization ratios computable (see X5, X6).

---

## Table 1: Properties

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| P1 | `subscription_plan_ordinal` | Free=0, Basic=1, Pro=2, Ultimate=3, Creator=4 | RET | Higher plan = more invested; Free tier is distinct from churned |
| P2 | `plan_monthly_credits` | Lookup: Basic=1000, Pro=3000, Ultimate=7000, Creator=20000, Free=300×30=9000 | — | Used as denominator in utilization features |
| P3 | `plan_monthly_cost_usd` | Lookup: Basic=9, Pro=29, Ultimate=49, Creator=149, Free=0 | — | Used in value-perception features |
| P4 | `tenure_days` | observation_date − subscription_start_date | BOTH | Short tenure = higher churn risk across types |
| P5 | `country_code_encoded` | Top 15 countries + "other" | BOTH | Geographic risk patterns |
| P6 | `subscription_start_month` | month(subscription_start_date) | BOTH | Cohort/seasonality effect |
| P7 | `subscription_start_dayofweek` | dayofweek(subscription_start_date) | VOL | Weekday signup = more intentional than weekend |

---

## Table 2: Generations (aggregated per user)

### 2a. Activity Volume

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G1 | `total_generations` | COUNT(generation_id) | RET | Core engagement volume |
| G2 | `n_completed` | COUNT WHERE status='completed' | RET | Successful value delivery |
| G3 | `n_failed` | COUNT WHERE status='failed' | BOTH | Platform reliability experience |
| G4 | `n_nsfw` | COUNT WHERE status='nsfw' | VOL | High NSFW rate = guardrail-testing, not serious creator; higher casual churn risk |
| G5 | `n_canceled` | COUNT WHERE status='canceled' | VOL | Active abandonment during generation |
| G6 | `n_queued_or_waiting` | COUNT WHERE status IN ('queued','waiting','in_progress') | INV | Stuck jobs at time of observation |

### 2b. Status Rates

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G7 | `completion_rate` | n_completed / total_generations | RET | Higher = better platform experience |
| G8 | `failure_rate_overall` | n_failed / total_generations | BOTH | Overall platform reliability |
| G9 | `failure_rate_last_10` | failures in last 10 gens / 10 | BOTH | Recent experience, leading indicator |
| G10 | `failure_rate_delta` | failure_rate_last_10 − failure_rate_overall | BOTH | Positive spike = deteriorating recent experience |
| G11 | `nsfw_rate` | n_nsfw / total_generations | VOL | Casual/exploratory user fingerprint |
| G12 | `cancellation_rate` | n_canceled / total_generations | VOL | Frustration with quality or speed |

### 2c. Credit / Monetization Usage

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G13 | `pct_free_model_usage` | COUNT(credit_cost IS NULL or 0) / total | VOL | Only free-model users have zero switching cost; NULL = free model per platform design |
| G14 | `n_credit_costing_gens` | COUNT WHERE credit_cost > 0 | RET | Committed paid-model engagement |
| G15 | `total_credits_consumed` | SUM(credit_cost) WHERE credit_cost > 0 | RET | Lifetime credit spend |
| G16 | `avg_credit_cost_per_gen` | MEAN(credit_cost) WHERE credit_cost > 0 | VOL | High avg cost = may feel value is insufficient |
| G17 | `credit_burn_rate` | total_credits_consumed / tenure_days | VOL | Rapid burnout → hitting credit ceiling |
| G18 | `credit_burn_acceleration` | credits_last_14d / (credits_prior_14d + 1) | VOL | Sudden drop = disengaging from paid models |
| G19 | `premium_gen_ratio` | COUNT(credit_cost > 30) / n_credit_costing_gens | RET | High-fidelity video usage = deepest platform commitment |

### 2d. Content Type Mix

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G20 | `video_generation_ratio` | COUNT(generation_type LIKE 'video_%') / total | RET | Video = higher investment, higher switching cost |
| G21 | `image_generation_ratio` | COUNT(generation_type LIKE 'image_%') / total | VOL | Image-only users have more free alternatives |
| G22 | `n_unique_generation_types` | COUNT DISTINCT(generation_type) | RET | Exploration breadth = embedded across platform |
| G23 | `video_to_image_graduation` | binary: first gen was image_*, later used video_* | RET | Deepening engagement trajectory |
| G24 | `dominant_generation_type_encoded` | MODE(generation_type) | BOTH | Usage pattern fingerprint |
| G25 | `image_model_1_share` | COUNT(image_model_1) / total | VOL | image_model_1 is the cheapest/most basic — casual-use proxy |
| G26 | `feature_expectation_mismatch` | binary: first_feature (quiz) contains 'video'/'Video' AND video_generation_ratio < 0.1 | VOL | Wanted video, only got images → unmet expectation |

### 2e. Quality Preferences

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G27 | `avg_video_duration` | MEAN(duration) WHERE video and duration IS NOT NULL | RET | Longer videos = higher cost tolerance and intent |
| G28 | `median_video_duration` | MEDIAN(duration) WHERE video | RET | Robust central tendency |
| G29 | `pct_high_resolution` | COUNT(resolution IN ('4k','2k')) / COUNT(resolution IS NOT NULL) | RET | Quality-seeking = invested user |
| G30 | `dominant_aspect_ratio_encoded` | MODE(aspect_ration) | BOTH | Usage fingerprint |

### 2f. Processing Quality (derived from timestamps)

> `processing_time_sec` exists only in test split. For train, derive as `completed_at − created_at` for `status='completed'`. Both methods produce equivalent values for completed generations.

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G31 | `avg_processing_time_sec` | MEAN(completed_at − created_at) WHERE completed | VOL | Long waits = QoS frustration risk |
| G32 | `median_processing_time_sec` | MEDIAN(same) | VOL | Robust QoS metric |
| G33 | `pct_long_wait_gens` | COUNT(processing_time > 300s) / n_completed | VOL | Frequency of hitting slow generation |

### 2g. Temporal Engagement

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G34 | `days_since_last_generation` | observation_date − MAX(created_at) | BOTH | RFM Recency — primary churn signal |
| G35 | `days_to_first_generation` | MIN(created_at) − subscription_start_date | BOTH | Slow activation → high churn risk; per research: 5-7 onboarding steps make this gap meaningful |
| G36 | `generation_span_days` | MAX(created_at) − MIN(created_at) | RET | Duration of active engagement window |
| G37 | `n_active_days` | COUNT DISTINCT(DATE(created_at)) | RET | Breadth of usage over time |
| G38 | `active_days_fraction` | n_active_days / tenure_days | RET | Utilization rate of subscription |
| G39 | `gens_first_7_days` | COUNT WHERE created_at ≤ subscription_start + 7d | RET | **Activation metric** — strongest early retention predictor in SaaS |
| G40 | `gens_last_7_days` | COUNT WHERE created_at ≥ observation_date − 7d | BOTH | Recent engagement level |
| G41 | `engagement_trajectory_ratio` | gens_second_half / (gens_first_half + 1) | VOL | <1.0 = declining; primary vol_churn leading indicator |
| G42 | `engagement_slope` | Linear regression coefficient on weekly generation counts | VOL | Trend direction; negative = fading |
| G43 | `gens_last_14_vs_prior_14` | COUNT(last 14d) / (COUNT(prior 14d) + 1) | VOL | Short-window momentum signal |

### 2h. Frequency and Session Patterns

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G44 | `avg_gens_per_active_day` | total_generations / n_active_days | RET | Intensity when engaged |
| G45 | `generation_frequency_daily` | total_generations / tenure_days | BOTH | RFM Frequency — overall platform usage rate |
| G46 | `generation_frequency_cv` | STDEV / MEAN of daily counts | BOTH | Bursty vs consistent; consistent users are stickier |
| G47 | `avg_inter_generation_hours` | MEAN(time between consecutive completed gens) | BOTH | Pace of work |
| G48 | `median_inter_generation_hours` | MEDIAN(same) | BOTH | Robust inter-event time |

### 2i. Time-of-Day / Day-of-Week Patterns

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G49 | `pct_gens_business_hours` | COUNT(hour(created_at) IN 9–18 AND weekday) / total | VOL | Professional use = business dependency = lower vol_churn |
| G50 | `pct_gens_weekdays` | COUNT(dayofweek(created_at) < 5) / total | VOL | Professional vs hobbyist signal |
| G51 | `dominant_usage_hour` | MODE(hour(created_at)) | BOTH | Usage time fingerprint |

---

## Table 3: Purchases (aggregated per user)

### 3a. Volume and Type

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU1 | `n_purchases_total` | COUNT(transaction_id) | RET | Overall successful payment activity |
| PU2 | `n_subscription_creates` | COUNT WHERE type='Subscription Create' | BOTH | Baseline subscription events |
| PU3 | `n_subscription_updates` | COUNT WHERE type='Subscription Update' | BOTH | Plan change activity |
| PU4 | `n_credit_package_purchases` | COUNT WHERE type='Credits package' | RET | **Strong retention signal** — hit credit ceiling, willingly paid more |
| PU5 | `n_upsell_purchases` | COUNT WHERE type='Upsell' | RET | Accepted upsell = high perceived platform value |
| PU6 | `has_reactivation_purchase` | binary: any type='Reactivation' | BOTH | Prior churn history in the record |
| PU7 | `pct_credit_package_purchases` | n_credit_package / n_total | RET | Share of deep-engagement purchases |

### 3b. Spend

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU8 | `total_purchase_dollars` | SUM(purchase_amount_dollars) | RET | Customer lifetime value to date |
| PU9 | `avg_purchase_dollars` | MEAN(purchase_amount_dollars) | RET | Willingness to pay signal |
| PU10 | `max_purchase_dollars` | MAX(purchase_amount_dollars) | RET | Peak spend; high = deeply committed |
| PU11 | `credit_package_spend_total` | SUM WHERE type='Credits package' | RET | Extra spend beyond subscription |

### 3c. Timing and Behavior

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU12 | `days_to_first_purchase` | MIN(purchase_time) − subscription_start_date | BOTH | Fast purchase = high intent; slow = trial behavior |
| PU13 | `generated_before_purchased` | binary: MIN(gen.created_at) < MIN(purchase_time) | VOL | Free trialer who later converted vs instant buyer |
| PU14 | `avg_days_between_purchases` | MEAN(diff between consecutive purchase_times) | BOTH | Regular payment cadence |
| PU15 | `days_since_last_purchase` | observation_date − MAX(purchase_time) | INV | Long gap = lapsed payment risk |
| PU16 | `has_plan_upgrade` | binary: Subscription Update with higher amount than previous | RET | Upgraded = satisfied and growing |
| PU17 | `has_plan_downgrade` | binary: Subscription Update with lower amount | VOL | Cost pressure or disengagement signal |
| PU18 | `n_plan_changes` | COUNT(Subscription Update) | BOTH | Plan instability = decision-making friction |

---

## Table 4: Transaction Attempts (aggregated per user)

> **Critical change:** `user_id` now exists directly in this table (col 3). All features below are computed by grouping directly on `user_id` — no longer require the intermediate join through purchases. This enables features T_NEW1–T_NEW4 which were previously impossible.

### 4a. NEW features enabled by user_id addition

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T_NEW1 | `has_failed_but_no_successful_payment` | binary: user has rows in transaction_attempts with failure_code IS NOT NULL but ZERO rows in purchases | INV | **Clearest invol_churn fingerprint** — tried to pay, never succeeded |
| T_NEW2 | `n_failed_without_matching_purchase` | COUNT failed txns WHERE transaction_id NOT IN purchases.transaction_id | INV | Scale of unresolved payment failures |
| T_NEW3 | `first_transaction_was_failure` | binary: MIN(transaction_time) has failure_code IS NOT NULL | INV | Payment broken from the very start — card issue at signup |
| T_NEW4 | `n_distinct_amounts_attempted` | COUNT DISTINCT(amount_in_usd) WHERE failure_code IS NOT NULL | INV | Trying different amounts = desperate retry behavior |

### 4b. Volume and Failure Rate

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T1 | `n_total_transaction_attempts` | COUNT per user_id | BOTH | Total payment events including failures |
| T2 | `n_successful_transactions` | COUNT WHERE failure_code IS NULL | RET | Smooth payment history |
| T3 | `n_failed_transactions` | COUNT WHERE failure_code IS NOT NULL | INV | Core involuntary churn signal |
| T4 | `transaction_failure_rate` | n_failed / n_total | INV | Payment reliability |
| T5 | `transaction_failure_rate_recent` | failures in last 30d / total_last_30d | INV | Escalating payment problems |
| T6 | `failure_rate_acceleration` | failure_rate_recent − failure_rate_overall | INV | Worsening payment situation signal |

### 4c. Failure Code Profile

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T7 | `n_card_declined` | COUNT WHERE failure_code='card_declined' | INV | Insufficient funds / card blocked |
| T8 | `n_cvc_failures` | COUNT WHERE failure_code IN ('incorrect_cvc','invalid_cvc') | INV | Card credential issue |
| T9 | `n_expired_card` | COUNT WHERE failure_code='expired_card' | INV | Card maintenance neglect |
| T10 | `n_auth_required_failures` | COUNT WHERE failure_code='authentication_required' | INV | 3D-secure setup incomplete |
| T11 | `n_processing_errors` | COUNT WHERE failure_code='processing_error' | INV | Platform-side failures |
| T12 | `dominant_failure_code_encoded` | MODE(failure_code) WHERE failure_code IS NOT NULL | INV | Primary payment failure mode |
| T13 | `payment_retry_count` | COUNT of failed txns within 24h of a prior failed txn, same user_id and same amount | INV | Retry = user trying to fix payment; cleaner now with user_id direct |

### 4d. Payment Instrument Risk

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T14 | `uses_prepaid_card` | binary: any is_prepaid=True | INV | Prepaid cards cannot support recurring billing |
| T15 | `uses_virtual_card` | binary: any is_virtual=True | INV | Virtual cards are often single-use |
| T16 | `uses_business_card` | binary: any is_business=True | RET | Business card = org-level purchase = higher retention |
| T17 | `uses_digital_wallet` | binary: any digital_wallet != 'none' | RET | Apple/Android Pay = frictionless = lower invol churn |
| T18 | `card_funding_type_encoded` | MODE(card_funding): debit/credit/prepaid/unknown | INV | Debit/prepaid = higher invol risk than credit |
| T19 | `pct_prepaid_transactions` | COUNT(is_prepaid=True) / n_total | INV | Proportion of risky payment attempts |

### 4e. Payment Friction Events

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T20 | `n_3d_secure_friction` | COUNT(is_3d_secure=True AND is_3d_secure_authenticated=False) | INV | Authentication wall hits |
| T21 | `has_any_cvc_failure` | binary | INV | One-shot payment credential error |
| T22 | `cvc_fail_rate` | COUNT(cvc_check='fail') / n_total | INV | Persistent credential issues |
| T23 | `n_cvc_unavailable` | COUNT(cvc_check='unavailable') | INV | Card doesn't support CVC — riskier instrument |

### 4f. Spend and Geography

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T24 | `total_transaction_amount_usd` | SUM(amount_in_usd) WHERE successful | RET | Total verified spend |
| T25 | `avg_transaction_amount_usd` | MEAN(amount_in_usd) | BOTH | Typical payment size |
| T26 | `n_high_value_transactions` | COUNT(amount_in_usd > 100) | RET | Premium spend events |
| T27 | `billing_matches_profile_country` | binary: normalize(billing_address_country) == normalize(country_code) | INV | Mismatch = VPN/expat = payment complication risk |
| T28 | `days_since_last_failed_transaction` | observation_date − MAX(transaction_time WHERE failed) | INV | Recency of payment problem |
| T29 | `days_since_last_successful_transaction` | observation_date − MAX(transaction_time WHERE successful) | INV | Last time payment actually worked |

---

## Table 5: Quizzes (per user)

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| Q1 | `quiz_completion_depth` | COUNT(non-null fields out of 8) | VOL | Intentionality proxy — low = casual user |
| Q2 | `acquisition_channel_intent` | source → high(google/chatgpt/ai-community)=2, mid(youtube/friends/twitter)=1, low(instagram/tiktok)=0 | VOL | Low-intent viral acquirees churn faster |
| Q3 | `source_encoded` | Top 10 categories + 'other' | BOTH | Channel fingerprint |
| Q4 | `team_size_ordinal` | solo/1=1, small=2, growing=3, 2-10=4, 11-50=5, large/enterprise=6; NaN=0 | VOL | Larger org = higher switching cost = lower vol_churn |
| Q5 | `experience_level_ordinal` | beginner=1, intermediate=2, advanced=3, expert=4; NaN=0 | VOL | Experts churn less from 'inconsistent' frustration |
| Q6 | `usage_plan_commercial` | binary: usage_plan IN (marketing, filmmaking, freelance, education, social) | VOL | Commercial dependency = lower vol_churn |
| Q7 | `usage_plan_encoded` | personal/social/filmmaking/marketing/freelance/education | VOL | Usage intent segment |
| Q8 | `frustrated_cost` | binary: frustration IN ('high-cost','High cost of top models') | VOL | Explicit cost-driven churn signal |
| Q9 | `frustrated_quality` | binary: frustration IN ('inconsistent','Inconsistent results') | VOL | Quality-driven; different intervention than cost |
| Q10 | `frustrated_limited` | binary: frustration IN ('limited','Limited generations') | VOL | Capacity-driven; may upgrade or leave |
| Q11 | `frustrated_confusing` | binary: frustration IN ('confusing','hard-prompt') | VOL | Onboarding/UX failure — early churn risk |
| Q12 | `first_feature_video` | binary: first_feature contains 'Video' or 'video' | RET | Video-first users have higher intent |
| Q13 | `first_feature_encoded` | top 8 categories + 'other' | BOTH | Platform entry-point fingerprint |
| Q14 | `role_commitment_score` | just-for-fun=0, creator/designer=1, marketer/educator=2, filmmaker/brand-owner=3, founder/prompt-engineer=4; NaN=0 | VOL | Higher professional stake = lower vol_churn |
| Q15 | `role_encoded` | top categories + 'other' | BOTH | Professional identity fingerprint |

---

## Cross-Table Derived Features

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| X1 | `payment_failure_timing_vs_activity` | MAX(gen.created_at) − MAX(failed txn.transaction_time) in days *(join changed)* | **KEY discriminator** | Positive = still generating when payment failed → INV; Negative = stopped generating, then payment lapsed → VOL |
| X2 | `active_during_payment_failure` | binary: had any generation within 7d of last failed transaction *(join changed)* | INV | Was the user engaged when payment broke? |
| X3 | `days_last_purchase_to_last_gen` | MAX(gen.created_at) − MAX(purchase.purchase_time) | VOL | Recently paid but stopped generating = early vol_churn signal |
| X4 | `spend_per_generation` | total_purchase_dollars / (total_generations + 1) | VOL | High cost-per-gen = value perception risk |
| X5 | `plan_credit_utilization_pct` | total_credits_consumed / (tenure_months × plan_monthly_credits) | BOTH | <20% = under-using, not getting value → VOL; consistently >100% = hitting limits → frustrated VOL or upgrade candidate |
| X6 | `plan_credit_surplus_deficit` | (tenure_months × plan_monthly_credits) − total_credits_consumed | BOTH | Negative = over-consuming; highly positive = leaving credits unused |
| X7 | `time_to_first_payment_issue` | MIN(failed txn.transaction_time) − subscription_start_date, days *(join changed)* | INV | How quickly after signup did payment problems emerge |
| X8 | `generation_to_purchase_ratio` | total_generations / (n_purchases_total + 1) | RET | Gens per transaction — value extraction efficiency |
| X9 | `is_likely_free_tier_user` | binary: n_purchases_total = 0 AND n_failed_transactions = 0 | — | No payment history at all = may be on free plan, not churned; exclude from invol_churn training signal |
| X10 | `activation_purchase_sequence` | MIN(gen.created_at) < MIN(purchase_time) → 1 (trial-to-paid), else 0 | VOL | Free trialer converted later vs. immediate buyer |
| X11 | `credit_per_dollar_spent` | total_credits_consumed / (total_purchase_dollars + 1) | VOL | Low credits per dollar = poor value perception |
| X12 | `n_failed_txns_before_first_success` | COUNT(failed txns WHERE transaction_time < MIN(successful txn time)) *(join changed)* | INV | How many failures before first successful payment |
| X13 | `failed_txn_to_gen_gap_days` | MIN(failed txn.transaction_time) − MIN(gen.created_at) *(join changed)* | INV | Negative = was generating before payment ever failed; positive = payment failed before first generation |

---

## Composite Scores

| # | Feature | Components | Purpose |
|---|---|---|---|
| CS1 | `payment_resilience_score` | T4 × w1 + T14 × w2 + T22 × w3 + T20 × w4 + T_NEW1 × w5 | Single invol_churn risk score; calibrate weights via logistic regression; explainable to PM |
| CS2 | `engagement_health_score` | G7 × w1 + G38 × w2 + G39_bool × w3 + G41 × w4 | Single behavioral health score for vol_churn risk |
| CS3 | `commitment_score` | Q4 + Q5 + Q6 + Q14 (normalized sum) | User's professional stake in the platform |
| CS4 | `rfm_recency_bin` | Quartile bucket of G34 | Standard RFM Recency component |
| CS5 | `rfm_frequency_bin` | Quartile bucket of G45 | Standard RFM Frequency component |
| CS6 | `rfm_monetary_bin` | Quartile bucket of PU8 | Standard RFM Monetary component |

---

## Features to Explicitly Drop

| Field | Reason |
|---|---|
| `flow_type` | 99.96% missing in train, 100% in test — zero variance |
| `card_country` (raw) | Use `billing_matches_profile_country` (T27) instead |
| `card_brand` (raw) | No meaningful churn signal beyond `card_funding` |
| `payment_method_type` | 100% 'card' — zero variance |
| `aspect_ration` (raw per-gen) | Use only `dominant_aspect_ratio_encoded` (G30) |
| `resolution` (raw per-gen) | Use only `pct_high_resolution` (G29) |
| `generation_type` (raw per-gen) | Use aggregated G20–G25 instead |
| `bank_name` | 2,683 unique values, high-cardinality noise |
| `duration` (raw per-gen) | Use summary features G27–G28 |
| `processing_time_sec` (test only) | Use derived G31–G33 for both splits; this column missing in train |

---

## Implementation Notes

**1. EDA summary is stale for transactions**  
The `.md` shows 19 columns. Actual files now have 20 (`user_id` at position 3). Re-run EDA notebook to update. All transaction aggregations should now group directly by `user_id` on the transaction_attempts table.

**2. Free tier identification (X9)**  
Users with zero purchases AND zero failed transactions are likely on the free plan (300 credits/day refresh, no billing required). These should NOT be labeled as invol_churn candidates. Flag them explicitly with `is_likely_free_tier_user` and consider excluding them from the Stage 2 (vol vs invol) classifier.

**3. Temporal leakage guard**  
All aggregations use only data prior to the churn observation date. For train, the observation date is the churn event date. Features like `failure_rate_recent` (last 30d) must be bounded to the observation window.

**4. Credit cost NULL semantics**  
Per platform design: `credit_cost IS NULL` = free model generation, not missing data. Set to 0 for aggregations. `credit_cost = 0` occurs for free-tier-accessible paid models. Both are distinct from `credit_cost > 0` (paid model usage).

**5. Test/train schema differences**  
- Test generations: has `processing_time_sec`; train does not → use derived G31–G33 for both  
- Test purchases: no 'Reactivation' type; train has 48 → `has_reactivation_purchase` is train-only signal  
- Subscription plan distribution shifts between splits: train has more Creator users; test has more Pro users  

**6. Two-stage model feature routing**  
- Stage 1 (churned vs not_churned): G-features, PU-features, Q-features, CS2, CS3 are most predictive  
- Stage 2 (vol vs invol, trained only on churned users): T-features, T_NEW features, CS1, X1, X2, X7, X12, X13 are most predictive  

**7. Zero-generation users**  
~4,384 train users have no generation rows. These default to 0 for all G-features. They likely cluster as invol_churn (never activated, payment failed at signup) or free tier. Cross-reference with T_NEW1 to separate these groups.

---

*Total features: ~135 before composite scores, ~141 with composites.*  
*Expected wide table: 90,000 rows × ~135 columns for train.*
