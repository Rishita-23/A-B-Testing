import numpy as np
from scipy import stats


def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """
    Compute the required sample size PER GROUP for a two-proportion z-test.
    Returns: - int : required sample size PER GROUP
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde

    # z-scores for the two-sided significance threshold and desired power
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided test
    z_beta = stats.norm.ppf(power)

    # pooled proportion and variance terms (standard formula for
    # two-proportion z-test sample size)
    p_bar = (p1 + p2) / 2
    variance_term = 2 * p_bar * (1 - p_bar)

    numerator = (z_alpha * np.sqrt(variance_term) + z_beta * np.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )) ** 2
    denominator = mde ** 2

    n = numerator / denominator
    return int(np.ceil(n))


def achieved_power(baseline_rate, mde, n_per_group, alpha=0.05):
    """
    Given a sample size you actually have (or plan to use) per group,
    compute the achieved statistical power
    Returns: -float : achieved power, between 0 and 1.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde

    p_bar = (p1 + p2) / 2
    se_pooled = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se_unpooled = np.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)

    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Under the alternative hypothesis (true effect = mde), how far is the
    # significance threshold from the true effect, in standard-error units?
    z = (mde - z_alpha * se_pooled) / se_unpooled
    power = stats.norm.cdf(z)
    return power


if __name__ == "__main__":
    BASELINE = 0.08
    MDE = 0.015
    ALPHA = 0.05
    POWER = 0.80

    n_needed = required_sample_size(BASELINE, MDE, alpha=ALPHA, power=POWER)
    print("POWER ANALYSIS: 'How many users do I need?'")
    print(f"  Baseline conversion rate : {BASELINE*100:.1f}%")
    print(f"  Minimum detectable effect: +{MDE*100:.1f} points")
    print(f"  Significance level (alpha): {ALPHA}")
    print(f"  Desired power             : {POWER*100:.0f}%")
    print(f"  --> Required sample size PER GROUP : {n_needed}")
    print(f"  --> Required TOTAL sample size      : {n_needed*2}")

    # Now check: with the sample size we actually simulated (5000/group),
    # what power did we actually have?
    ACTUAL_N = 5000
    p = achieved_power(BASELINE, MDE, ACTUAL_N, alpha=ALPHA)
    print(f"  With our actual simulated n={ACTUAL_N} per group,")
    print(f"  achieved power = {p*100:.1f}%")
