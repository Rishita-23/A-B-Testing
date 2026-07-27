import numpy as np
from scipy import stats

# PART 1 Beta Binomial posterior
def compute_posterior(conversions, n, prior_alpha=1, prior_beta=1):
    """
    Compute the Beta posterior distribution for a group's true conversion
    rate, given observed data and a Beta(prior_alpha, prior_beta) prior.
 
    Returns: -
    (posterior_alpha, posterior_beta) : tuple of floats defining the
    posterior Beta distribution.
    """
    non_conversions = n - conversions
    posterior_alpha = prior_alpha + conversions
    posterior_beta = prior_beta + non_conversions
    return posterior_alpha, posterior_beta

def posterior_summary(posterior_alpha, posterior_beta, credible_mass=0.95):
    """Summarize a Beta posterior: mean, and a credible interval"""

    mean = posterior_alpha / (posterior_alpha + posterior_beta)
    lower = stats.beta.ppf((1 - credible_mass) / 2, posterior_alpha, posterior_beta)
    upper = stats.beta.ppf((1 + credible_mass) / 2, posterior_alpha, posterior_beta)
    return{"mean" : mean, "ci_lower" : lower, "ci_upper": upper}

def probability_treatment_beats_control(
        control_alpha, control_beta, treatment_alpha, treatment_beta,n_samples= 200_000, seed = 42
):
    """
    Estimate P(treatment_rate > control_rate) via Monte Carlo sampling:
    draw many random samples from each posterior distribution, and check
    how often the treatment sample beats the control sample.
    """
    rng  = np.random.default_rng(seed)
    control_samples = rng.beta(control_alpha, control_beta, size=n_samples)
    treatment_samples = rng.beta(treatment_alpha, treatment_beta, size= n_samples)

    prob_treatment_better = np.mean(treatment_samples > control_samples)
    uplift_sample = treatment_samples - control_samples

    return{
        "prob_treatment_better" : prob_treatment_better,
        "expected_uplift": np.mean(uplift_sample),
        "uplift_ci_lower": np.percentile(uplift_sample, 2.5),
        "uplift_ci_upper": np.percentile(uplift_sample, 97.5),
        "uplift_samples" : uplift_sample, # returned plotting in the dashboard
        "control_samples": control_samples,
        "treatment_samples": treatment_samples,
    }

# Part 2 Comparision 
def compare_frequentist_and_bayesian(conversion_a, n_a, conversion_b, n_b, alpha=0.05):
    from frequentist_testing import two_proportion_ztest

    freq_result = two_proportion_ztest(conversion_a, n_a, conversion_b, n_b, alpha=alpha)

    control_post_a, control_post_b = compute_posterior(conversion_a, n_a)
    treatment_post_a, treatment_post_b = compute_posterior(conversion_b, n_b)

    control_summary = posterior_summary(control_post_a, control_post_b)
    treatment_summary = posterior_summary(treatment_post_a, treatment_post_b)

    bayes_result = probability_treatment_beats_control(
        control_post_a, control_post_b, treatment_post_a, treatment_post_b
    )

    bayes_confident=(
        bayes_result["prob_treatment_better"] > 0.95
        or bayes_result["prob_treatment_better"] < 0.05 
    )
    agree = freq_result["significant"] == bayes_confident

    if agree:
        agreement_text =(
            "Both methods AGREE: "
            + ("both find a confident/significant effect." if freq_result["significant"]
               else "neither finds strong enough evidence of an effect yet.")
        )
    else: 
        agreement_text =(
            "The methods land in DIFFERENT places, this can happen near " \
            "the edges of significant/confidence, and is a good discussion " \
            "point about how much each framework treats uncertainty different."
        )

    print("SIDE-BY-SIDE: Frequentist vs Bayesian")
    print("\n")
    print("FREQUENTIST: ")
    print(f" p-value = {freq_result['p_value']:.4f} | significant at alpha={alpha}? {freq_result['significant']}")
    print(f" {freq_result['summary']}")
    print(f"\nBAYESIAN")
    print(f" P(treatment > control) = {bayes_result['prob_treatment_better']*100:.1f}%")
    print(f" Expected uplift = {bayes_result['expected_uplift']*100:+.2f}pts")
    print(f" 95% credible interval = [{bayes_result['uplift_ci_lower']*100:+.2f},{bayes_result['uplift_ci_upper']*100:+.2f}%]")
    print(f" Control posterior mean = {control_summary['mean']*100:.2f}% (95%CrI : [{control_summary['ci_lower']*100:.2f}, {control_summary['ci_upper']*100:.2f}%])")
    print(f" Treatment posterior mean = {treatment_summary['mean']*100:.2f}% (95% CrI: [{treatment_summary['ci_lower']*100:.2f}, {treatment_summary['ci_upper']*100:.2f}])")
    print("\n" + agreement_text)

    return{
        "frequentist" : freq_result,
        "bayesian" : bayes_result,
        "control_posterior_summary" : control_summary,
        "treatment_posterior_summary": treatment_summary,
        "control_posterior_params" : (control_post_a, control_post_b),
        "treatment_posterior_params" : (treatment_post_a, treatment_post_b),
        "agree" : agree,
        "agreement_text" : agreement_text,
    }

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/experiment_data.csv")
    control = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]

    compare_frequentist_and_bayesian(
        control["conversion"].sum(), len(control),
        treatment["conversion"].sum(), len(treatment),
    )