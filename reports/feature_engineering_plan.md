# Feature Engineering Plan
## Higgsfield Churn Prediction — Complete Feature Specification

**Labels:** `not_churned` / `vol_churn` / `invol_churn`  
**Grain:** one row per `user_id`  
**Signal tags:** `VOL` = voluntary churn signal, `INV` = involuntary churn signal, `RET` = retention signal, `BOTH` = separates churned from not-churned regardless of type

---

## Table 1: Properties

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| P1 | `subscription_plan_ordinal` | Basic=1, Creator=2, Pro=3, Ultimate=4 | RET | Higher plan = more invested |
| P2 | `tenure_days` | observation_date − subscription_start_date | BOTH | Short tenure = higher churn risk across types |
| P3 | `country_code_encoded` | Top 15 countries + "other" | BOTH | Geographic risk patterns |
| P4 | `subscription_start_month` | month(subscription_start_date) | BOTH | Cohort/seasonality effect |
| P5 | `subscription_start_dayofweek` | dayofweek(subscription_start_date) | VOL | Weekday signup = more intentional than weekend |

---

## Table 2: Generations (aggregated per user)

### 2a. Activity Volume

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G1 | `total_generations` | COUNT(generation_id) | RET | Core engagement volume |
| G2 | `n_completed` | COUNT WHERE status='completed' | RET | Successful value delivery |
| G3 | `n_failed` | COUNT WHERE status='failed' | BOTH | Platform reliability experience |
| G4 | `n_nsfw` | COUNT WHERE status='nsfw' | BOTH | Distinct behavioral segment |
| G5 | `n_canceled` | COUNT WHERE status='canceled' | VOL | Active abandonment during generation |
| G6 | `n_queued_or_waiting` | COUNT WHERE status IN ('queued','waiting','in_progress') | INV | Stuck jobs at time of observation |

### 2b. Status Rates

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G7 | `completion_rate` | n_completed / total_generations | RET | Higher = better platform experience |
| G8 | `failure_rate_overall` | n_failed / total_generations | BOTH | Overall platform reliability |
| G9 | `failure_rate_last_10` | failures in last 10 gens / 10 | BOTH | Recent experience, leading indicator |
| G10 | `failure_rate_delta` | failure_rate_last_10 − failure_rate_overall | BOTH | Spike = deteriorating experience |
| G11 | `nsfw_rate` | n_nsfw / total_generations | BOTH | Behavioral fingerprint |
| G12 | `cancellation_rate` | n_canceled / total_generations | VOL | Frustration with generation quality/speed |

### 2c. Credit / Monetization Usage

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G13 | `pct_free_model_usage` | COUNT(credit_cost IS NULL or 0) / total | VOL | Users only on free models have zero switching cost |
| G14 | `n_credit_costing_gens` | COUNT WHERE credit_cost > 0 | RET | Committed usage — paid model engagement |
| G15 | `total_credits_consumed` | SUM(credit_cost) WHERE credit_cost > 0 | RET | Lifetime credit spend proxy |
| G16 | `avg_credit_cost_per_gen` | MEAN(credit_cost) WHERE credit_cost > 0 | VOL | Perception of value; high cost = frustration risk |
| G17 | `credit_burn_rate` | total_credits_consumed / tenure_days | VOL | Rapid burnout = may feel credits insufficient |
| G18 | `credit_burn_acceleration` | credits_last_14d / credits_prior_14d | VOL | Sudden drop = disengaging from paid models |

### 2d. Content Type Mix

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G19 | `video_generation_ratio` | COUNT(video_model_*) / total | RET | Video = higher investment, higher switching cost |
| G20 | `image_generation_ratio` | COUNT(image_model_*) / total | VOL | Image-only users have more free alternatives |
| G21 | `n_unique_generation_types` | COUNT DISTINCT(generation_type) | RET | Exploration breadth = embedded in platform |
| G22 | `video_to_image_graduation` | binary: first gen was image, later used video | RET | Deepening engagement trajectory |
| G23 | `dominant_generation_type_encoded` | MODE(generation_type) | BOTH | Usage pattern fingerprint |
| G24 | `image_model_1_share` | COUNT(image_model_1) / total | BOTH | Heavy model_1 usage = lowest cost / most casual |

### 2e. Quality Preferences

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G25 | `avg_video_duration` | MEAN(duration) WHERE video and duration IS NOT NULL | RET | Longer duration = higher cost tolerance |
| G26 | `median_video_duration` | MEDIAN(duration) WHERE video | RET | Robust central tendency |
| G27 | `pct_high_resolution` | COUNT(resolution IN ('4k','2k')) / COUNT(resolution IS NOT NULL) | RET | Quality-seeking behavior |
| G28 | `dominant_aspect_ratio_encoded` | MODE(aspect_ration) | BOTH | Usage fingerprint |

### 2f. Processing Quality (derived from timestamps)

> Note: train set lacks `processing_time_sec`; compute as `completed_at − created_at` for `status='completed'`

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G29 | `avg_processing_time_sec` | MEAN(completed_at − created_at) WHERE completed | VOL | Long waits = frustration risk |
| G30 | `median_processing_time_sec` | MEDIAN(same) | VOL | Robust QoS metric |
| G31 | `pct_long_wait_gens` | COUNT(processing_time > 300s) / n_completed | VOL | How often user hit slow generation |

### 2g. Temporal Engagement (critical group)

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G32 | `days_since_last_generation` | observation_date − MAX(created_at) | BOTH | RFM Recency |
| G33 | `days_to_first_generation` | MIN(created_at) − subscription_start_date | BOTH | Slow activation = high churn risk |
| G34 | `generation_span_days` | MAX(created_at) − MIN(created_at) | RET | Engagement duration within platform |
| G35 | `n_active_days` | COUNT DISTINCT(DATE(created_at)) | RET | Breadth of usage over time |
| G36 | `active_days_fraction` | n_active_days / tenure_days | RET | Utilization rate of subscription |
| G37 | `gens_first_7_days` | COUNT WHERE created_at ≤ subscription_start + 7d | RET | **Activation metric** — strongest early retention signal |
| G38 | `gens_last_7_days` | COUNT WHERE created_at ≥ observation_date − 7d | BOTH | Recent engagement level |
| G39 | `gens_first_half` | COUNT WHERE created_at in first half of generation_span | BOTH | Used in trajectory calculation |
| G40 | `gens_second_half` | COUNT WHERE created_at in second half of generation_span | BOTH | Used in trajectory calculation |
| G41 | `engagement_trajectory_ratio` | gens_second_half / (gens_first_half + 1) | VOL | <1 = declining; primary vol_churn leading indicator |
| G42 | `engagement_slope` | Linear regression coef on weekly generation counts | VOL | Trend direction; negative = fading engagement |
| G43 | `gens_last_14_vs_prior_14` | COUNT(last 14d) / COUNT(prior 14d) | VOL | Short-window momentum signal |

### 2h. Frequency and Session Patterns

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G44 | `avg_gens_per_active_day` | total_generations / n_active_days | RET | Intensity when engaged |
| G45 | `generation_frequency_daily` | total_generations / tenure_days | BOTH | Overall platform usage rate |
| G46 | `generation_frequency_cv` | STDEV / MEAN of daily counts | BOTH | Bursty vs consistent users; consistent = more embedded |
| G47 | `avg_inter_generation_hours` | MEAN(time between consecutive completed gens) | BOTH | Pace of work |
| G48 | `median_inter_generation_hours` | MEDIAN(same) | BOTH | Robust inter-event time |

### 2i. Time-of-Day / Day-of-Week Patterns

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| G49 | `pct_gens_business_hours` | COUNT(hour(created_at) IN 9–18 AND weekday) / total | VOL | Professional use = business dependency = lower vol churn |
| G50 | `pct_gens_weekdays` | COUNT(dayofweek(created_at) < 5) / total | VOL | Professional vs hobbyist signal |
| G51 | `dominant_usage_hour` | MODE(hour(created_at)) | BOTH | Usage time fingerprint |

---

## Table 3: Purchases (aggregated per user)

### 3a. Volume and Type

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU1 | `n_purchases_total` | COUNT(transaction_id) | RET | Overall transaction engagement |
| PU2 | `n_subscription_creates` | COUNT WHERE type='Subscription Create' | BOTH | Baseline subscription events |
| PU3 | `n_subscription_updates` | COUNT WHERE type='Subscription Update' | BOTH | Plan change activity |
| PU4 | `n_credit_package_purchases` | COUNT WHERE type='Credits package' | RET | **Strong retention signal** — hit limits, willingly paid more |
| PU5 | `n_upsell_purchases` | COUNT WHERE type='Upsell' | RET | Accepted upsell = high perceived value |
| PU6 | `has_reactivation_purchase` | binary: any type='Reactivation' | BOTH | Prior churn history — known churner archetype |
| PU7 | `pct_credit_package_purchases` | n_credit_package / n_total | RET | Ratio of deep-engagement purchases |

### 3b. Spend

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU8 | `total_purchase_dollars` | SUM(purchase_amount_dollars) | RET | Customer lifetime value to date |
| PU9 | `avg_purchase_dollars` | MEAN(purchase_amount_dollars) | RET | Willingness to pay |
| PU10 | `max_purchase_dollars` | MAX(purchase_amount_dollars) | RET | Peak spend event; high = deeply committed |
| PU11 | `credit_package_spend_total` | SUM WHERE type='Credits package' | RET | Extra spend beyond subscription |

### 3c. Timing and Behavior

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| PU12 | `days_to_first_purchase` | MIN(purchase_time) − subscription_start_date | BOTH | Fast purchase = high intent |
| PU13 | `generated_before_purchased` | binary: MIN(gen.created_at) < MIN(purchase_time) | VOL | Free trialer who later converted vs instant buyer |
| PU14 | `avg_days_between_purchases` | MEAN(diff between consecutive purchase_times) | BOTH | Regular payment cadence |
| PU15 | `days_since_last_purchase` | observation_date − MAX(purchase_time) | INV | Long gap = lapsed payment risk |
| PU16 | `has_plan_upgrade` | binary: Subscription Update with higher amount | RET | Upgraded = satisfied |
| PU17 | `has_plan_downgrade` | binary: Subscription Update with lower amount | VOL | Downgraded = cost pressure or disengagement |
| PU18 | `n_plan_changes` | COUNT(Subscription Update) | BOTH | Plan instability = decision-making friction |

---

## Table 4: Transactions (aggregated per user)

> Note: `transaction_id` in this table links to `transaction_id` in purchases. Failed transactions have no entry in purchases.

### 4a. Volume and Failure Rate

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T1 | `n_total_transaction_attempts` | COUNT(transaction_id) | BOTH | Total payment events |
| T2 | `n_successful_transactions` | COUNT WHERE failure_code IS NULL | RET | Smooth payment history |
| T3 | `n_failed_transactions` | COUNT WHERE failure_code IS NOT NULL | INV | Core involuntary churn signal |
| T4 | `transaction_failure_rate` | n_failed / n_total | INV | Payment reliability |
| T5 | `transaction_failure_rate_recent` | failures in last 30d / total_last_30d | INV | Escalating payment problems |
| T6 | `failure_rate_acceleration` | failure_rate_recent − failure_rate_overall | INV | Worsening payment situation |

### 4b. Failure Code Profile

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T7 | `n_card_declined` | COUNT WHERE failure_code='card_declined' | INV | Insufficient funds / card blocked |
| T8 | `n_cvc_failures` | COUNT WHERE failure_code IN ('incorrect_cvc','invalid_cvc') | INV | Card credential issue |
| T9 | `n_expired_card` | COUNT WHERE failure_code='expired_card' | INV | Card maintenance neglect |
| T10 | `n_auth_required_failures` | COUNT WHERE failure_code='authentication_required' | INV | 3D-secure setup incomplete |
| T11 | `n_processing_errors` | COUNT WHERE failure_code='processing_error' | INV | Platform-side failures |
| T12 | `dominant_failure_code_encoded` | MODE(failure_code) WHERE failure_code IS NOT NULL | INV | Primary payment failure mode |
| T13 | `payment_retry_count` | COUNT of failed txns within 24h of another failed txn (same user) | INV | User is trying to fix payment — high invol_churn intent |

### 4c. Payment Instrument Risk

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T14 | `uses_prepaid_card` | binary: any is_prepaid=True | INV | Prepaid cards = no recurring billing possible |
| T15 | `uses_virtual_card` | binary: any is_virtual=True | INV | Virtual cards often single-use |
| T16 | `uses_business_card` | binary: any is_business=True | RET | Business card = org-level purchase = higher retention |
| T17 | `uses_digital_wallet` | binary: any digital_wallet != 'none' | RET | Apple/Android Pay = frictionless payment = lower invol churn |
| T18 | `card_funding_type_encoded` | MODE(card_funding): debit/credit/prepaid/unknown | INV | Debit/prepaid = higher invol risk than credit |
| T19 | `pct_prepaid_transactions` | COUNT(is_prepaid=True) / n_total | INV | Proportion of risky payment attempts |

### 4d. Payment Friction Events

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T20 | `n_3d_secure_friction` | COUNT(is_3d_secure=True AND is_3d_secure_authenticated=False) | INV | Authentication wall hits |
| T21 | `has_any_cvc_failure` | binary | INV | One-shot payment credential error |
| T22 | `cvc_fail_rate` | COUNT(cvc_check='fail') / n_total | INV | Persistent credential issues |
| T23 | `n_cvc_unavailable` | COUNT(cvc_check='unavailable') | INV | Card doesn't support CVC — higher risk instrument |

### 4e. Spend and Geography

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| T24 | `total_transaction_amount_usd` | SUM(amount_in_usd) WHERE successful | RET | Total verified spend |
| T25 | `avg_transaction_amount_usd` | MEAN(amount_in_usd) | BOTH | Typical payment size |
| T26 | `n_high_value_transactions` | COUNT(amount_in_usd > 100) | RET | Premium spend events |
| T27 | `billing_matches_profile_country` | binary: normalize(billing_address_country) == normalize(country_code) | INV | Mismatch = VPN/expat = payment complication risk |
| T28 | `days_since_last_failed_transaction` | observation_date − MAX(transaction_time WHERE failed) | INV | Recency of payment problem |
| T29 | `days_since_last_successful_transaction` | observation_date − MAX(transaction_time WHERE successful) | INV | Last time payment actually worked |

---

## Table 5: Quizzes (per user, mostly categorical encoding)

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| Q1 | `quiz_completion_depth` | COUNT(non-null fields) out of 8 | VOL | Intentionality proxy; low = casual, high = purposeful |
| Q2 | `acquisition_channel_intent` | source → high(google/chatgpt/ai-community)=2, mid(youtube/friends/twitter)=1, low(instagram/tiktok)=0 | VOL | Low-intent viral acquirees churn faster |
| Q3 | `source_encoded` | Top 10 categories + 'other' | BOTH | Channel fingerprint |
| Q4 | `team_size_ordinal` | solo/1=1, small=2, growing=3, 2-10=4, 11-50=5, 2001-5000+=6, enterprise=7; NaN=0 | VOL | Larger org = higher switching cost = lower vol_churn |
| Q5 | `experience_level_ordinal` | beginner=1, intermediate=2, advanced=3, expert=4; NaN=0 | VOL | Experts churn less from "inconsistent" frustration |
| Q6 | `usage_plan_commercial` | binary: usage_plan IN (marketing, filmmaking, freelance, education, social) | VOL | Commercial use = business dependency = lower vol_churn |
| Q7 | `usage_plan_encoded` | personal/social/filmmaking/marketing/freelance/education | VOL | Usage intent segment |
| Q8 | `frustrated_cost` | binary: frustration IN ('high-cost','High cost of top models') | VOL | Explicit cost-driven churn signal |
| Q9 | `frustrated_quality` | binary: frustration IN ('inconsistent','Inconsistent results') | VOL | Quality-driven churn; different intervention needed |
| Q10 | `frustrated_limited` | binary: frustration IN ('limited','Limited generations') | VOL | Capacity-driven; may upgrade or leave |
| Q11 | `frustrated_confusing` | binary: frustration IN ('confusing','hard-prompt') | VOL | Onboarding/UX failure signal |
| Q12 | `first_feature_video` | binary: first_feature contains 'Video' or 'video' | RET | Video-first users have higher intent |
| Q13 | `first_feature_encoded` | top 8 categories + 'other' | BOTH | Platform entry-point fingerprint |
| Q14 | `role_commitment_score` | just-for-fun=0, creator/designer=1, marketer/educator=2, filmmaker/brand-owner=3, founder/prompt-engineer=4; NaN=0 | VOL | Higher professional stake = lower vol_churn |
| Q15 | `role_encoded` | top categories + 'other' | BOTH | Professional identity fingerprint |

---

## Cross-Table Derived Features (require joining multiple tables)

| # | Feature | Derivation | Signal | Notes |
|---|---|---|---|---|
| X1 | `payment_failure_timing_vs_activity` | MAX(gen.created_at) − MAX(failed txn.transaction_time) in days | **KEY vol/invol discriminator** | Positive = still active when payment failed (INV); Negative = stopped generating then payment lapsed (VOL) |
| X2 | `active_during_payment_failure` | binary: had any generation within 7d of last failed transaction | INV | Was the user engaged when payment broke? |
| X3 | `days_last_purchase_to_last_gen` | MAX(gen.created_at) − MAX(purchase.purchase_time) | VOL | Recently paid but stopped using = early vol_churn signal |
| X4 | `spend_per_generation` | total_purchase_dollars / total_generations | VOL | High cost per generation = value perception risk |
| X5 | `plan_usage_alignment` | subscription_plan_ordinal − NTILE(4)(generation_frequency_daily) | BOTH | Negative = over-paying for actual usage; positive = hitting limits |
| X6 | `activation_purchase_sequence` | MIN(gen.created_at) relative to MIN(purchase.purchase_time) | VOL | Generated before first payment = trial behavior vs immediate buyer |
| X7 | `time_to_first_payment_issue` | MIN(failed txn.transaction_time) − subscription_start_date | INV | How quickly payment problems emerged |
| X8 | `generation_to_purchase_ratio` | total_generations / n_purchases_total | RET | Gens per dollar spent — value extraction efficiency |
| X9 | `credit_package_trigger_depth` | total_credits_consumed / (subscription tier credit allowance) | VOL | How hard user is hitting credit limits; proxy for plan-feature fit |

---

## Composite Scores (model-ready engineered features)

| # | Feature | Components | Purpose |
|---|---|---|---|
| CS1 | `payment_resilience_score` | T4 × w1 + T14 × w2 + T22 × w3 + T20 × w4 (calibrate via logistic reg) | Single invol_churn risk score; explainable to PM |
| CS2 | `engagement_health_score` | G7 × w1 + G36 × w2 + G37_binary × w3 + G41 × w4 | Single behavioral health score for vol_churn risk |
| CS3 | `commitment_score` | Q4 + Q5 + Q6 + Q14 (normalized sum) | User's professional stake in the platform |
| CS4 | `rfm_recency` | Binned G32 (days since last gen) into quartiles | Standard RFM component |
| CS5 | `rfm_frequency` | Binned G45 (gens/day) into quartiles | Standard RFM component |
| CS6 | `rfm_monetary` | Binned PU8 (total spend) into quartiles | Standard RFM component |

---

## Features to Explicitly Drop

| Field | Reason |
|---|---|
| `flow_type` | 99.96% missing in train, 100% missing in test |
| `card_country` (raw) | Use billing_matches_profile_country (X7) instead |
| `card_brand` (raw) | No meaningful churn signal beyond card_funding |
| `payment_method_type` | 100% 'card' — zero variance |
| `aspect_ration` (raw) | Keep only dominant_aspect_ratio_encoded (G28) |
| `resolution` (raw per-gen) | Keep only pct_high_resolution (G27) |
| `generation_type` (raw per-gen) | Keep aggregated features G19–G24 |
| `bank_name` | 2683 unique values, high cardinality noise |
| `duration` (raw per-gen) | Keep avg/median summary features G25–G26 |

---

## Implementation Notes

1. **Temporal leakage guard**: All generation/transaction aggregations must use only data prior to the churn event date. For training, the churn event is the label; use all available data up to the observation window.

2. **Test set schema difference**: Test generations include `processing_time_sec` which train lacks. Use derived `completed_at − created_at` for train (G29–G31). Both methods should agree closely for completed generations.

3. **Missing value strategy**:
   - Quiz fields: NaN → encode as explicit "unknown" category or 0 in ordinal scales. Do NOT drop rows.
   - `credit_cost` NULL: per idea #2 (confirmed), NULL = free model. Treat as 0 credits consumed.
   - Users with zero generations: valid — 4,384 train users have no generation rows. Features default to 0 / NaN as appropriate. These users may cluster into a pure invol_churn group (never activated, payment failed).

4. **Credit cost imputation**: For `credit_cost IS NULL`, distinguish between free-model usage (generation_type known to be free) and missing data. Per EDA and idea #2, `credit_cost=NULL` on free models is correct — not missing data.

5. **Two-stage modeling note**: Features T1–T29 and CS1 are primarily useful in a Stage 2 classifier (vol vs invol, trained on churned users only). Features G1–G51, Q1–Q15, and CS2–CS3 are primarily useful in Stage 1 (churned vs not_churned).

---

*Total features: ~120 before composite scores, ~126 with composites.*  
*Expected wide table dimensions: 90,000 rows × ~120 columns for train.*
