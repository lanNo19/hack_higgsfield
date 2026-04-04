# Higgsfield Platform Research: Subscription & User Journey

This document provides a breakdown of Higgsfield's business logic and user flow to assist in feature engineering for the user retention model.

## 1. Subscription Tiers & Credit Allocation (2026)

Higgsfield utilizes a credit-based economy. Users on higher tiers receive a monthly allotment of credits and gain access to more compute-intensive models (e.g., Sora 2, Veo 3.1).

| Plan | Monthly Cost (USD) | Monthly Credits | Key Model Access | Concurrent Jobs |
| :--- | :--- | :--- | :--- | :--- |
| **Free** | $0 | 300 (Renewed daily) | Basic Video/Image | 1 |
| **Basic** | ~$9.00 | 1,000 | Seedream 4.0 | 2 |
| **Pro** | ~$29.00 | 3,000 | Veo 3.1, Sora 2 | 4 |
| **Ultimate** | ~$49.00 | 7,000 | All Models + 4K | 6 |
| **Creator** | ~$149.00 | 20,000 | Priority Access | 10+ |

### Generation Economics
To calculate the **Burn Rate** feature, use the following estimated costs per generation:
* **Standard Image:** 1–5 Credits
* **High-Res Image (4K):** 10–15 Credits
* **Basic Video (5s):** 20–30 Credits
* **High-Fidelity Video (Veo/Sora):** 50–100 Credits

---

## 2. Onboarding Flow: "Path to First Value"

The platform requires a user to navigate approximately **5 to 7 steps** before they can generate their first image/video. This is a critical window for churn analysis.

1.  **Landing Page:** Value proposition.
2.  **Authentication:** Google/Apple/Email Sign-up.
3.  **The Onboarding Quiz:** (Source for `test_users_quizzes.csv`)
    * Users select their role (Creator, Marketer, Hobbyist).
    * Users state their experience level and primary feature interest.
4.  **Paywall/Plan Selection:** High-friction point where users choose a tier or continue with "Free."
5.  **Main Dashboard:** The "Project Selection" screen.
6.  **Creation Interface:** Setting aspect ratio, prompt, and model.
7.  **Generation Queue:** The actual "Submit" action.

---

## 3. Mapping Research to CSV Data

### `test_users_generations.csv`
* **`credit_cost`:** Used to distinguish between "Free" usage and "Premium" model usage.
* **`status`:** Look for `failed` or `nsfw`. A high ratio of `nsfw` flags suggests the user is testing the guardrails and may not be a long-term "serious" creator.
* **`duration`:** High-duration videos consume more tokens; users generating long videos are likely "Pro" or "Creator" tier.

### `test_users_quizzes.csv`
* **`frustration`:** Direct sentiment analysis. High frustration at onboarding is a strong predictor of Day-1 churn.
* **`first_feature`:** If a user wants "Video" but only uses "Image" models, there is a feature-mismatch that could lead to churn.

### `test_users_purchases.csv` / `test_users_properties.csv`
* **Conversion Lag:** Time delta between `subscription_start_date` and the first `purchase_time`.
* **Plan Upgrades:** Identify users who move from "Basic" to "Creator" to flag "High-Value" clusters.

---

## 4. Key Metrics for Feature Engineering

* **Credit Velocity:** $\frac{\text{Total Credits Spent}}{\text{Days Active}}$
* **Generation Efficiency:** $\frac{\text{Completed Generations}}{\text{Total Attempts}}$
* **Onboarding Speed:** Time from `user_id` creation to first `generation_id`.
* **Diversity Score:** Unique count of `generation_type` and `resolution` per user.