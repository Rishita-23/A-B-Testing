import numpy as np
import pandas as pd

# CONFIGURABLE PARAMETERS (change these to simulate different scenarios)
RANDOM_SEED = 42              # makes results reproducible
N_PER_VARIANT = 5000          # number of users in EACH of control/treatment
BASELINE_CONVERSION_RATE = 0.08   # 8% of control users convert (buy something)
TRUE_TREATMENT_EFFECT = 0.015     # true lift: treatment converts 1.5 percentage
                                   # points higher than control (i.e. 9.5%)
N_DAYS = 14                    # the experiment "runs" for 14 days
OUTPUT_PATH = "data/experiment_data.csv"


def simulate_experiment(
    n_per_variant=N_PER_VARIANT,
    baseline_rate=BASELINE_CONVERSION_RATE,
    true_effect=TRUE_TREATMENT_EFFECT,
    n_days=N_DAYS,
    seed=RANDOM_SEED,
):
    """
    Simulate a two-group (control vs treatment) A/B test.
    Returns: - pandas.DataFrame with columns: user_id, variant, day, conversion
    """
    rng = np.random.default_rng(seed)

    treatment_rate = baseline_rate + true_effect

    # Assign users to groups
    n_total = n_per_variant * 2
    variants = np.array(["control"] * n_per_variant + ["treatment"] * n_per_variant)

    # Assign each user a day (1 through n_days), spread evenly 
    # We use np.tile/repeat-style spreading so each day gets a roughly equal
    # number of control and treatment users, like a real running experiment.
    days_per_variant = rng.integers(1, n_days + 1, size=n_total)

    # Simulate conversion outcomes
    # For each user, conversion is a coin flip (Bernoulli trial) with a
    # probability equal to their group's true conversion rate.
    conversion_probs = np.where(variants == "control", baseline_rate, treatment_rate)
    conversions = rng.binomial(n=1, p=conversion_probs)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_total + 1),
            "variant": variants,
            "day": days_per_variant,
            "conversion": conversions,
        }
    )

    # Shuffle rows so it looks like a realistic event log, not grouped by variant
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def print_summary(df, baseline_rate, true_effect):
    print("SIMULATED A/B TEST: Checkout Page Redesign")
    for variant in ["control", "treatment"]:
        subset = df[df["variant"] == variant]
        n = len(subset)
        conv_rate = subset["conversion"].mean()
        print(f"  {variant.upper():10s} | n = {n:5d} | observed conversion rate = {conv_rate:.4f} ({conv_rate*100:.2f}%)")

    observed_lift = (
        df[df["variant"] == "treatment"]["conversion"].mean()
        - df[df["variant"] == "control"]["conversion"].mean()
    )

    print(f"  TRUE baseline rate injected      : {baseline_rate:.4f} ({baseline_rate*100:.2f}%)")
    print(f"  TRUE treatment effect injected   : +{true_effect:.4f} (+{true_effect*100:.2f} pts)")
    print(f"  OBSERVED lift in this sample      : {observed_lift:+.4f} ({observed_lift*100:+.2f} pts)")
    print(f"  Days simulated                    : {df['day'].nunique()}")
    print(f"  Total users                        : {len(df)}")
    print("Note: observed lift won't exactly equal the true effect because")
    print("of random sampling noise -- that's expected and is exactly what")
    print("statistical testing (next steps) is designed to deal with.")


if __name__ == "__main__":
    import os

    df = simulate_experiment()

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print_summary(df, BASELINE_CONVERSION_RATE, TRUE_TREATMENT_EFFECT)
    print(f"\nSaved to {OUTPUT_PATH}")
