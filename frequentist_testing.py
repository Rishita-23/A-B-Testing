import numpy as np 
import pandas as pd
from scipy import stats

# PART 1 Standard two proportion z-test
def two_proportion_ztest(conversions_a, n_a, conversions_b, n_b, alpha=0.05):
    """
    Run a two-proportion z-test comparing group A (e.g. control) to
    group B (e.g. treatment). 
    Return:- 
        dict with:
        rate_a, rate_b       : observed conversion rates
        effect               : rate_b - rate_a (absolute difference)
        z_stat               : z statistic
        p_value              : two-sided p-value
        ci_lower, ci_upper   : confidence interval on the effect (rate_b - rate_a)
        significant          : bool, whether p_value < alpha
        summary              : plain-English sentence describing the result
    """

    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b
    effect = rate_b - rate_a

    # Pooled proportion, used for the ztest statistics
    # (standard approach under the null hypothesis that both groups have the SAME true rate)
    pooled_rate = (conversions_a + conversions_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled_rate *(1- pooled_rate) * (1 / n_a +1 / n_b))

    if se_pooled == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = effect / se_pooled
        p_value = 2*(1-stats.norm.cdf(abs(z_stat))) #two sided

    # Confidence interval uses the UNPOOLED standard error
    # (standard approach for CIs on the difference between two proportions)
    se_unpooled = np.sqrt(rate_a *(1-rate_a) / n_a + rate_b *(1- rate_b)/ n_b)
    z_crit = stats.norm.ppf(1-alpha / 2)
    ci_upper = effect + z_crit * se_unpooled
    ci_lower = effect - z_crit * se_unpooled

    significant = p_value < alpha

    if significant:
        direction = "higher" if effect > 0 else "lower"
        summary =(
            f"Treatment conversion rate ({rate_b*100:.2f}%) is statistically "
            f"significantly {direction} than control ({rate_a*100:.2f}%) at"
            f"alpha = {alpha}. p-value = {p_value:.4f}."
        )
    else:
        summary=(
            f"No statistically significant difference detected between"
            f" treatment ({rate_b*100:.2f}%) and control ({rate_a*100:.2f}%)"
            f"at alpha = {alpha}, p-value = {p_value:.4f}. This does NOT prove"
            f"there's no effect -- it means we don't have strong enough"
            f"evidence yet."
        )
    return{
        "rate_a" : rate_a,
        "rate_b" : rate_b,
        "effect" : effect,
        "z_stat" : z_stat,
        "p_value" : p_value,
        "ci_lower" : ci_lower,
        "ci_upper" : ci_upper,
        "significant" : significant,
        "summary" : summary,
    }

# PART 2
def simulate_peeking_problem(
        baseline_rate = 0.08,
        n_per_day_per_group = 350,
        n_days = 14,
        n_simulation = 1000,
        alpha = 0.05,
        seed =123,
):
    """
    Simulate many fake A/B tests where there is deliberately NO true
    treatment effect (control and treatment have the exact same true
    conversion rate), and measure how often each stopping strategy
    incorrectly declares a "significant" result (a false positive).
 
    Returns: -
    dict with:
        fixed_horizon_fpr : false-positive rate using the "only look once,
                             at the end" strategy (should be close to alpha)
        peeking_fpr       : false-positive rate using the "check every day,
                             stop at first significant result" strategy
                             (expected to be noticeably HIGHER than alpha)
        daily_pvalues_example : list of p-values from ONE example simulated
                             run, for plotting purposes
    """
    rng = np.random.default_rng(seed)

    fixed_horizon_false_positives = 0
    peeking_false_positives = 0
    example_daily_pvalues = None

    for sim in range(n_simulation):
        # Simulate n_days worth of daily conversion counts for both groups
        # using the SAME true rate for both (no real effects exists)
        daily_conversions_a = rng.binomial(n_per_day_per_group, baseline_rate, size=n_days)
        daily_conversions_b = rng.binomial(n_per_day_per_group, baseline_rate, size=n_days)

        cumulative_n_a = 0
        cumulative_n_b = 0
        cumulative_conv_a = 0
        cumulative_conv_b = 0

        daily_pvalues =[]
        peeked_significant_early = False

        for day in range(n_days):
            cumulative_n_a += n_per_day_per_group
            cumulative_n_b += n_per_day_per_group
            cumulative_conv_a += daily_conversions_a[day]
            cumulative_conv_b += daily_conversions_b[day]

            result = two_proportion_ztest(
                cumulative_conv_a, cumulative_n_a, cumulative_conv_b,
                cumulative_n_b, alpha = alpha
            )
            daily_pvalues.append(result["p_value"])

            # 'Peeking' strategy: as soon as ANY day is significant, a 
            # peeker would stop and declare a (false) win. We only need to
            # know if this happened at least once during the run
            if result["significant"] and not peeked_significant_early:
                peeked_significant_early = True

        # Fixed horizon strategy: only the FINAL day's p-values counts
        final_day_significant  = daily_pvalues[-1] < alpha

        if final_day_significant:
            fixed_horizon_false_positives += 1
        if peeked_significant_early:
            peeking_false_positives += 1

        if sim == 0:
            example_daily_pvalues = daily_pvalues
 
    fixed_horizon_fpr = fixed_horizon_false_positives / n_simulation
    peeking_fpr = peeking_false_positives / n_simulation 

    return{
        "fixed_horizon_fpr" : fixed_horizon_fpr,
        "peeking_fpr" : peeking_fpr,
        "n_simulations" : n_simulation,
        "alpha" : alpha,
        "daily_pvalues_example" : example_daily_pvalues,
        "n_days" : n_days,
    }

if __name__ == "__main__":
    print("PART 1: Two proportion z-test on our simulated experiment")

    df = pd.read_csv("data/experiment_data.csv")
    control = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]

    result = two_proportion_ztest(
        control["conversion"].sum(), len(control),
        treatment["conversion"].sum(), len(treatment),
    )

    print(f" Control Rate :  {result['rate_a']*100:.2f}%")
    print(f" Treatment rate : {result['rate_b']*100:.2f}%")
    print(f" Effect(abs)    : {result['effect']*100:.2f}%")
    print(f" 95% CI on effect : [{result['ci_lower']*100:+.2f}, {result['ci_upper']*100:+.2f}] pts")
    print(f" z-statistics   : {result['z_stat']:.3f}")
    print(f" p-value    : {result['p_value']:.6f}")
    print(f" Significant at alpha = 0.05? {result['significant']}")
    print(f" Plain English : {result['summary']}")

    print("\n" + "-"*20)
    print("PART 2: The Peeking Problem (500 simulated experiments, No true effect)")
    peek_result = simulate_peeking_problem(
        baseline_rate = 0.08,
        n_per_day_per_group = 350,
        n_days = 14,
        n_simulation = 1000,
        alpha = 0.05,
        seed = 123,
    )
    

    print(f"\n Ran {peek_result['n_simulations']} simulated experiments, each {peek_result['n_days']} days,")
    print(f" with control and treatment given the IDENTICAL true conversion rate")
    print(f" (so ANY 'significant' result found is by definition a false positive).")
    print(f" Expaected false-positive rate(alpha) : {peek_result['alpha']*100:.1f}%")
    print(f"ACTUAL false-positive rate, DAILY PEEKING : {peek_result['fixed_horizon_fpr']*100:.1f}%")
    print("\n")
    inflation_factor = peek_result['peeking_fpr'] / peek_result['alpha']
    print(f" --> Peeking daily inflate your false-positive rate to about")
    print(f"  {inflation_factor:.1f}x the nominal 5% you thought you were getting")
