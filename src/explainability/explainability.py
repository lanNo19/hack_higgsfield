"""
Explainability Engine for Higgsfield Churn Prediction (HackNU 2026)
===================================================================

Designed for Pipeline 2 output: a single Optuna-tuned GBDT model
(LightGBM, XGBoost, or CatBoost).

Produces:
  1. Global feature importance per churn type (vol vs invol)
  2. Per-user waterfall explanations with business-language annotations
  3. Intervention strategy mapping (feature cluster → recommended action)
  4. Export-ready figures for presentation

Usage:
    python explainability.py \
        --model model.pkl \
        --features features_train.csv \
        --holdout-frac 0.15 \
        --out-dir ./explainability_output

    OR pass pre-split data:
    python explainability.py \
        --model model.pkl \
        --X-holdout X_holdout.csv \
        --y-holdout y_holdout.csv \
        --out-dir ./explainability_output
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# =====================================================================
# BUSINESS TRANSLATION LAYER
# =====================================================================

FEATURE_DISPLAY_NAMES = {
    # Subscription
    "sub_tenure_days": "Account age (days)",
    "plan_tier": "Subscription plan tier",
    "is_top_tier_plan": "On Ultimate plan",
    "is_bottom_tier_plan": "On Creator/Basic plan",
    "country_freq": "Country prevalence",
    "is_rare_country": "From rare country",
    "sub_is_weekend": "Signed up on weekend",
    "sub_hour": "Signup hour of day",
    "sub_dow": "Signup day of week",
    # Purchases
    "total_purchases": "Total purchases made",
    "total_spend": "Total amount spent ($)",
    "avg_purchase_amount": "Average purchase amount ($)",
    "std_purchase_amount": "Purchase amount variability",
    "max_purchase_amount": "Largest single purchase ($)",
    "min_purchase_amount": "Smallest purchase ($)",
    "spend_cv": "Spending inconsistency (CV)",
    "spend_range": "Spending range ($)",
    "days_since_last_purchase": "Days since last purchase",
    "days_since_first_purchase": "Days since first purchase",
    "purchase_frequency": "Purchase frequency",
    "n_purch_credits_package": "Credit package purchases",
    "n_purch_subscription_create": "Subscription creations",
    "n_purch_subscription_update": "Subscription updates",
    "n_purch_upsell": "Upsell purchases",
    "n_purch_gift": "Gift purchases",
    "has_credits_purchase": "Bought extra credits",
    "has_upsell": "Accepted an upsell offer",
    "has_gift": "Made a gift purchase",
    "mean_interpurchase_days": "Avg days between purchases",
    "std_interpurchase_days": "Purchase interval variability",
    # Transactions / Billing
    "total_txn_attempts": "Total payment attempts",
    "n_failed_txn": "Failed payment attempts",
    "n_success_txn": "Successful payments",
    "total_txn_amount": "Total transaction amount ($)",
    "avg_txn_amount": "Average transaction amount ($)",
    "max_txn_amount": "Largest transaction ($)",
    "txn_failure_rate": "Payment failure rate",
    "has_any_failure": "Had any payment failure",
    "all_failed": "All payments failed",
    "multiple_cards_used": "Tried multiple cards",
    "n_unique_cards": "Number of cards used",
    "days_since_last_txn": "Days since last transaction",
    "days_since_first_txn": "Days since first transaction",
    "max_fail_streak": "Max consecutive payment failures",
    "last_txn_failed": "Last transaction failed",
    "n_fail_card_declined": "Card declined count",
    "n_fail_expired_card": "Expired card failures",
    "n_fail_incorrect_cvc": "Incorrect CVC failures",
    "n_fail_incorrect_number": "Incorrect card number failures",
    "n_fail_processing_error": "Processing errors",
    "n_fail_authentication_required": "Authentication required failures",
    "n_fail_invalid_cvc": "Invalid CVC failures",
    "any_prepaid": "Uses prepaid card",
    "any_virtual": "Uses virtual card",
    "any_business": "Uses business card",
    "n_cvc_fail": "CVC check failures",
    "n_cvc_not_provided": "CVC not provided count",
    "any_3ds_required": "3D Secure required",
    "n_3ds_attempted": "3D Secure attempts",
    "n_3ds_authenticated": "3D Secure authenticated",
    "_3ds_fail_count": "3D Secure failures",
    "country_mismatch_rate": "Billing/card country mismatch rate",
    "uses_digital_wallet": "Uses digital wallet",
    "card_funding_debit": "Primary card is debit",
    "card_funding_credit": "Primary card is credit",
    "card_funding_prepaid": "Primary card is prepaid",
    "card_brand_visa": "Card brand: Visa",
    "card_brand_mc": "Card brand: Mastercard",
    "card_brand_amex": "Card brand: Amex",
    # Generations / Engagement
    "total_generations": "Total video generations",
    "n_completed_gens": "Completed generations",
    "n_failed_gens": "Failed generations",
    "n_nsfw_gens": "NSFW-flagged generations",
    "total_credits_spent": "Total credits consumed",
    "avg_credits_per_gen": "Avg credits per generation",
    "n_unique_gen_types": "Model types used",
    "gen_success_rate": "Generation success rate",
    "gen_failure_rate": "Generation failure rate",
    "nsfw_rate": "NSFW flagging rate",
    "days_since_last_gen": "Days since last generation",
    "days_since_first_gen": "Days since first generation",
    "gen_frequency_per_day": "Generations per day",
    "credit_burn_rate": "Credits consumed per day",
    "gen_type_entropy": "Generation type diversity",
    "gen_engagement_trend": "Engagement trend (rising/falling)",
    "mean_intergen_hours": "Avg hours between generations",
    "std_intergen_hours": "Generation interval variability",
    "max_intergen_hours": "Longest gap between generations (hrs)",
    "gen_gap_trend": "Generation gaps increasing",
    # Quiz / Onboarding
    "quiz_completion_score": "Onboarding quiz completeness",
    "quiz_fully_complete": "Quiz fully completed",
    "quiz_empty": "Quiz left empty",
    "source_is_organic_social": "Found us via social media",
    "source_is_community": "Found us via community/friends",
    "source_is_chatgpt": "Found us via ChatGPT",
    "source_is_google": "Found us via Google search",
    "source_filled": "Provided acquisition source",
    "experience_level": "Self-reported experience level",
    "is_beginner": "Self-reported beginner",
    "is_expert_or_advanced": "Self-reported expert/advanced",
    "is_solo_user": "Solo user",
    "is_team_user": "Team user",
    "usage_is_professional": "Professional use case",
    "usage_is_personal": "Personal use case",
    "frust_is_cost": "Frustrated by cost",
    "frust_is_inconsistent": "Frustrated by inconsistent results",
    "frust_is_limited": "Frustrated by limited generations",
    "frust_is_hard_prompt": "Frustrated by prompt difficulty",
    "frustration_filled": "Reported a frustration",
    "ff_is_video": "Interested in video generation",
    "ff_is_commercial": "Interested in commercial/ad videos",
    "ff_is_avatar": "Interested in avatars/lipsync",
    "ff_is_image": "Interested in image editing",
    "role_is_creator": "Role: content creator",
    "role_is_professional": "Role: professional (filmmaker/designer/marketer)",
    # Cross-table interactions
    "engaged_but_failing_pay": "Active user with payment issues",
    "paying_but_not_using": "Paying but not generating content",
    "good_onboarding_low_use": "Completed onboarding but low usage",
    "overpaying_for_tier": "High plan tier but low usage",
    "new_user_payment_fail": "New user with payment failure",
    "cost_frustrated_high_spender": "Cost-frustrated high spender",
    "expert_on_basic": "Expert user on basic plan",
    "credits_per_dollar": "Credits obtained per dollar spent",
}

# Intervention recommendations per feature cluster
INTERVENTION_MAP = {
    # Involuntary churn interventions
    "billing_failure": {
        "features": ["txn_failure_rate", "n_failed_txn", "max_fail_streak",
                     "last_txn_failed", "all_failed", "has_any_failure",
                     "n_fail_card_declined", "engaged_but_failing_pay"],
        "label": "Payment Recovery",
        "action": "Implement smart retry logic with exponential backoff. "
                  "Send pre-dunning email 3 days before expected charge. "
                  "Offer alternative payment methods (digital wallet, different card).",
    },
    "card_issues": {
        "features": ["n_fail_expired_card", "n_fail_incorrect_cvc", "n_fail_invalid_cvc",
                     "n_fail_incorrect_number", "any_prepaid", "any_virtual",
                     "multiple_cards_used", "n_unique_cards"],
        "label": "Card Update Campaign",
        "action": "Trigger card update reminder when expiry approaches. "
                  "Enable account updater service for automatic card refresh. "
                  "Flag prepaid/virtual card users for proactive outreach.",
    },
    "3ds_friction": {
        "features": ["_3ds_fail_count", "any_3ds_required", "n_3ds_attempted",
                     "country_mismatch_rate"],
        "label": "Authentication Friction Reduction",
        "action": "Optimize 3D Secure flow (request exemptions where possible). "
                  "Ensure billing country matches card country. "
                  "Consider region-specific payment processors.",
    },
    # Voluntary churn interventions
    "engagement_decay": {
        "features": ["days_since_last_gen", "gen_frequency_per_day", "gen_gap_trend",
                     "gen_engagement_trend", "total_generations", "paying_but_not_using",
                     "mean_intergen_hours", "max_intergen_hours"],
        "label": "Re-engagement Campaign",
        "action": "Send personalized 'what's new' email with new model/feature highlights. "
                  "Offer free credits to restart generation activity. "
                  "Show gallery of trending community creations for inspiration.",
    },
    "value_perception": {
        "features": ["frust_is_cost", "overpaying_for_tier", "cost_frustrated_high_spender",
                     "plan_tier", "is_top_tier_plan", "expert_on_basic"],
        "label": "Value Optimization",
        "action": "For overpaying users: suggest downgrade to right-sized plan (retain > lose). "
                  "For cost-frustrated users: offer limited-time discount or annual plan savings. "
                  "For experts on basic: offer trial upgrade to demonstrate premium value.",
    },
    "onboarding_gap": {
        "features": ["quiz_empty", "quiz_completion_score", "good_onboarding_low_use",
                     "is_beginner", "frust_is_hard_prompt"],
        "label": "Onboarding Improvement",
        "action": "Trigger guided tutorial sequence for users with incomplete onboarding. "
                  "Provide prompt templates for beginners struggling with prompt engineering. "
                  "Assign onboarding specialist for high-tier plan users who haven't generated.",
    },
    "low_adoption": {
        "features": ["n_unique_gen_types", "gen_type_entropy", "has_credits_purchase",
                     "has_upsell", "total_credits_spent"],
        "label": "Feature Adoption Drive",
        "action": "Send targeted feature discovery emails based on unused capabilities. "
                  "Offer credit bundles for trying new generation models. "
                  "Highlight ROI stories from similar users who expanded usage.",
    },
}

CLASS_NAMES = {0: "not_churned", 1: "vol_churn", 2: "invol_churn"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12"}

# Which intervention clusters belong to which churn class
# 1 = vol_churn, 2 = invol_churn
CLUSTER_CLASS_MAP = {
    "billing_failure":  2,
    "card_issues":      2,
    "3ds_friction":     2,
    "engagement_decay": 1,
    "value_perception": 1,
    "onboarding_gap":   1,
    "low_adoption":     1,
}


# =====================================================================
# HELPERS
# =====================================================================

def get_display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def load_model(path: Path):
    """Load model. Supports CatBoost native (.cbm), joblib/pickle (.pkl)."""
    path = Path(path)
    if path.suffix == ".cbm":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    else:
        import joblib
        return joblib.load(path)


def detect_model_type(model) -> str:
    cls = type(model).__name__.lower()
    if "lgbm" in cls or "lightgbm" in cls:
        return "lightgbm"
    elif "xgb" in cls:
        return "xgboost"
    elif "catboost" in cls:
        return "catboost"
    else:
        return "unknown"


# =====================================================================
# 1. GLOBAL EXPLANATIONS
# =====================================================================

def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using TreeExplainer."""
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values shape depends on model type:
    # - LightGBM multiclass: list of 3 arrays, each (n_samples, n_features)
    # - XGBoost multiclass: (n_samples, n_features, 3)
    # - CatBoost multiclass: list of 3 arrays

    # Normalize to shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    elif shap_values.ndim == 3 and shap_values.shape[2] == 3:
        pass  # already correct
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    print(f"  SHAP values shape: {shap_values.shape}")
    return explainer, shap_values


def plot_global_importance(shap_values, X, class_idx, class_name, out_dir, top_n=15):
    """SHAP summary plot for one class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sv = shap_values[:, :, class_idx]
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]

    display_names = [get_display_name(X.columns[i]) for i in top_idx]

    shap.summary_plot(
        sv[:, top_idx],
        X.iloc[:, top_idx],
        feature_names=display_names,
        plot_type="dot",
        show=False,
        max_display=top_n,
    )
    plt.title(f"Top {top_n} Features Driving {class_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"shap_global_{class_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")
    return top_idx


def plot_global_bar_comparison(shap_values, X, out_dir, top_n=15):
    """Side-by-side bar chart: vol_churn vs invol_churn top features."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, cls_idx, cls_name, color in [
        (axes[0], 1, "Voluntary Churn", "#e74c3c"),
        (axes[1], 2, "Involuntary Churn", "#f39c12"),
    ]:
        sv = shap_values[:, :, cls_idx]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]

        names = [get_display_name(X.columns[i]) for i in top_idx]
        values = mean_abs[top_idx]

        ax.barh(range(top_n), values[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(cls_name, fontsize=13, fontweight="bold")

    plt.suptitle(
        "What Drives Each Churn Type? (Feature Importance Comparison)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = out_dir / "shap_comparison_vol_vs_invol.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# =====================================================================
# 2. PER-USER EXPLANATIONS
# =====================================================================

def explain_user(
        shap_values, X, user_idx, predicted_class, out_dir, explainer, tag=""
):
    """Generate waterfall plot + text explanation for one user."""
    cls_idx = predicted_class
    cls_name = CLASS_NAMES[cls_idx]
    sv = shap_values[user_idx, :, cls_idx]

    # Build shap Explanation object for waterfall
    explanation = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value[cls_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X.iloc[user_idx].values,
        feature_names=[get_display_name(c) for c in X.columns],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(
        f"Why This User Was Flagged: {cls_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fname = f"shap_waterfall_{tag}_{cls_name}.png"
    path = out_dir / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    # Text explanation
    top_k = 5
    top_feat_idx = np.argsort(np.abs(sv))[-top_k:][::-1]
    lines = [f"--- User Explanation ({cls_name}) ---"]
    for i in top_feat_idx:
        feat = X.columns[i]
        display = get_display_name(feat)
        val = X.iloc[user_idx, i]
        contribution = sv[i]
        direction = "increases" if contribution > 0 else "decreases"
        lines.append(
            f"  • {display} = {val:.4g} → {direction} {cls_name} risk "
            f"(SHAP: {contribution:+.4f})"
        )

    # Map to intervention — pass cls_idx so only class-appropriate clusters are considered
    intervention = map_to_intervention(top_feat_idx, X.columns, cls_idx)
    if intervention:
        lines.append(f"\n  Recommended intervention: {intervention['label']}")
        lines.append(f"  Action: {intervention['action']}")

    explanation_text = "\n".join(lines)
    print(explanation_text)

    text_path = out_dir / f"explanation_{tag}_{cls_name}.txt"
    text_path.write_text(explanation_text)

    return explanation_text


def map_to_intervention(top_feat_indices, columns, predicted_class: int):
    """Map top SHAP features to the best-matching intervention cluster.

    Only considers clusters that belong to the predicted churn type
    (invol clusters for class 2, vol clusters for class 1), preventing
    payment-recovery interventions from being assigned to vol_churn users
    and vice versa.
    """
    top_feats = set(columns[i] for i in top_feat_indices)
    best_match = None
    best_overlap = 0
    for cluster_key, cluster in INTERVENTION_MAP.items():
        if CLUSTER_CLASS_MAP.get(cluster_key) != predicted_class:
            continue
        overlap = len(top_feats & set(cluster["features"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = cluster
    # Fallback: no feature overlap found — return first cluster for this class
    if best_match is None:
        for cluster_key, cluster in INTERVENTION_MAP.items():
            if CLUSTER_CLASS_MAP.get(cluster_key) == predicted_class:
                return cluster
    return best_match


def select_example_users(shap_values, y, predicted_probs, n_per_class=2):
    """Select high-confidence example users for vol_churn and invol_churn only.

    not_churned users are excluded — there is no intervention to recommend
    for retained users and they add noise to the presentation output.
    """
    examples = {}
    for cls_idx in [1, 2]:  # vol_churn, invol_churn only
        cls_name = CLASS_NAMES[cls_idx]
        mask = y == cls_idx
        if mask.sum() == 0:
            continue
        probs_for_class = predicted_probs[mask, cls_idx]
        local_indices = np.where(mask)[0]
        top_local = np.argsort(probs_for_class)[-n_per_class:][::-1]
        examples[cls_name] = local_indices[top_local]

    return examples


# =====================================================================
# 3. STRATEGY SUMMARY
# =====================================================================

def generate_strategy_summary(shap_values, X, out_dir, top_n=10):
    """
    Produce a structured JSON + text summary mapping:
      churn type → top features → intervention cluster → recommended action
    """
    summary = {}
    for cls_idx, cls_name in [(1, "vol_churn"), (2, "invol_churn")]:
        sv = shap_values[:, :, cls_idx]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]

        features_ranked = []
        for i in top_idx:
            feat = X.columns[i]
            features_ranked.append({
                "feature": feat,
                "display_name": get_display_name(feat),
                "mean_abs_shap": float(mean_abs[i]),
            })

        # Find matching interventions
        top_feats = set(X.columns[i] for i in top_idx)
        matched_interventions = []
        for cluster_key, cluster in INTERVENTION_MAP.items():
            overlap = top_feats & set(cluster["features"])
            if overlap:
                matched_interventions.append({
                    "cluster": cluster_key,
                    "label": cluster["label"],
                    "matched_features": sorted(overlap),
                    "action": cluster["action"],
                })

        summary[cls_name] = {
            "top_features": features_ranked,
            "interventions": matched_interventions,
        }

    # Save JSON
    json_path = out_dir / "strategy_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path.name}")

    # Save readable text
    text_lines = ["=" * 70, "  CHURN INTERVENTION STRATEGY SUMMARY", "=" * 70, ""]
    for cls_name, data in summary.items():
        text_lines.append(f"{'─' * 50}")
        text_lines.append(f"  {cls_name.replace('_', ' ').upper()}")
        text_lines.append(f"{'─' * 50}")
        text_lines.append("")
        text_lines.append("  Top drivers:")
        for i, f in enumerate(data["top_features"][:7], 1):
            text_lines.append(f"    {i}. {f['display_name']} (SHAP: {f['mean_abs_shap']:.4f})")
        text_lines.append("")
        text_lines.append("  Recommended interventions:")
        for intv in data["interventions"]:
            text_lines.append(f"    ▸ {intv['label']}")
            text_lines.append(f"      {intv['action']}")
            text_lines.append(f"      (Driven by: {', '.join(get_display_name(f) for f in intv['matched_features'])})")
            text_lines.append("")
        text_lines.append("")

    text = "\n".join(text_lines)
    text_path = out_dir / "strategy_summary.txt"
    text_path.write_text(text)
    print(f"  Saved: {text_path.name}")

    return summary


# =====================================================================
# 4. PRESENTATION-READY FIGURE
# =====================================================================

def plot_churn_risk_distribution(predicted_probs, y, out_dir):
    """Histogram of churn probabilities colored by true class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cls_idx, cls_name, color in [
        (axes[0], 1, "Voluntary Churn Risk", "#e74c3c"),
        (axes[1], 2, "Involuntary Churn Risk", "#f39c12"),
    ]:
        for true_cls in [0, 1, 2]:
            mask = y == true_cls
            ax.hist(
                predicted_probs[mask, cls_idx],
                bins=50, alpha=0.5, label=CLASS_NAMES[true_cls],
                color=CLASS_COLORS[true_cls],
            )
        ax.set_xlabel(f"Predicted P({cls_name})", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(cls_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("Churn Risk Score Distribution by True Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "churn_risk_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# =====================================================================
# MAIN
# =====================================================================

def run_explainability(
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        predicted_probs: np.ndarray | None = None,
        out_dir: Path = Path("./explainability_output"),
):
    """
    Full explainability pipeline.

    Args:
        model: trained GBDT model (LightGBM/XGBoost/CatBoost)
        X: feature matrix (holdout set)
        y: true labels (0=not_churned, 1=vol_churn, 2=invol_churn)
        predicted_probs: model.predict_proba(X), shape (n, 3). Computed if None.
        out_dir: directory for output figures and text
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_type = detect_model_type(model)
    print(f"\n{'='*60}")
    print(f"  Explainability Pipeline — {model_type.upper()}")
    print(f"  Holdout samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"{'='*60}\n")

    # Get predictions if not provided
    if predicted_probs is None:
        print("  Computing predictions...")
        predicted_probs = model.predict_proba(X)

    # --- SHAP ---
    explainer, shap_values = compute_shap_values(model, X)

    # --- Global importance plots ---
    print("\n  [1/4] Global feature importance per churn type...")
    for cls_idx, cls_name in [(1, "vol_churn"), (2, "invol_churn")]:
        plot_global_importance(shap_values, X, cls_idx, cls_name, out_dir)

    plot_global_bar_comparison(shap_values, X, out_dir)

    # --- Per-user waterfall explanations ---
    print("\n  [2/4] Per-user waterfall explanations...")
    examples = select_example_users(shap_values, y, predicted_probs, n_per_class=2)
    all_explanations = []

    for cls_name, indices in examples.items():
        for rank, idx in enumerate(indices):
            # Use predicted class for SHAP explanation, not the true label.
            # Explaining against the true class when the model disagrees produces
            # near-zero SHAP values because the model's uncertainty is spread
            # across classes — we want to explain what the model actually decided.
            cls_idx = int(np.argmax(predicted_probs[idx]))
            tag = f"{cls_name}_example{rank+1}"
            explanation = explain_user(
                shap_values, X, idx, cls_idx, out_dir, explainer, tag=tag
            )
            all_explanations.append(explanation)
            print()

    # --- Strategy summary ---
    print("  [3/4] Generating intervention strategy summary...")
    generate_strategy_summary(shap_values, X, out_dir)

    # --- Risk distribution ---
    print("\n  [4/4] Plotting churn risk distributions...")
    plot_churn_risk_distribution(predicted_probs, y, out_dir)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {out_dir}")
    print(f"{'='*60}\n")

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "examples": examples,
        "explanations": all_explanations,
    }


# =====================================================================
# CLI
# =====================================================================

def run(out_dir: Path | None = None) -> dict:
    """Convenience entry point using repo defaults (best CatBoost model + holdout split)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.models.pipeline_utils import load_train_data, make_holdout
    from src.utils.helpers import root_path

    model_path = root_path() / "models" / "trained" / "catboost_final.cbm"
    if out_dir is None:
        out_dir = root_path() / "explainability_output"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: uv run python train_final_model.py"
        )

    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    print("Loading training data...")
    X, y = load_train_data()
    _, X_hold, _, y_hold = make_holdout(X, y)
    X_hold = X_hold.reset_index(drop=True)

    # Align features to model's training features
    train_features = model.feature_names_
    if train_features is not None:
        missing = [f for f in train_features if f not in X_hold.columns]
        for f in missing:
            X_hold[f] = 0
        X_hold = X_hold[train_features]

    return run_explainability(model, X_hold, y_hold, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability for Higgsfield Churn Model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model (.cbm or .pkl). Default: models/trained/catboost_final.cbm")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to feature parquet (uses repo default if omitted)")
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    # Use convenience runner if no custom args given
    if args.model is None and args.features is None:
        run(out_dir=Path(args.out_dir) if args.out_dir else None)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.utils.helpers import processed_path, root_path
        from src.models.pipeline_utils import LABEL_MAP

        model_path = Path(args.model) if args.model else root_path() / "models" / "trained" / "catboost_final.cbm"
        model = load_model(model_path)

        if args.features:
            df = pd.read_parquet(args.features) if args.features.endswith(".parquet") else pd.read_csv(args.features)
        else:
            df = pd.read_parquet(processed_path() / "features_train.parquet")

        if "churn_status" in df.columns:
            y_full = df["churn_status"].map(LABEL_MAP).values
        else:
            raise ValueError("No churn_status column in features file")

        X_full = df.drop(columns=["user_id", "churn_status"], errors="ignore")

        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(
            X_full, y_full,
            test_size=args.holdout_frac,
            stratify=y_full,
            random_state=42,
        )
        X = X.reset_index(drop=True)

        out_dir = Path(args.out_dir) if args.out_dir else root_path() / "explainability_output"
        run_explainability(model, X, y, out_dir=out_dir)