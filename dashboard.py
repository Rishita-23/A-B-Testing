import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from power_analysis import required_sample_size, achieved_power
from frequentist_testing import two_proportion_ztest,simulate_peeking_problem
from bayesian_testing import(
    compute_posterior,
    posterior_summary,
    probability_treatment_beats_control,
) 

st.set_page_config(page_title= "A/B Testing & Experiment Simulator", layout = "wide")
DATA_PATH = "data/experiment_data.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def run_peeking_simulation(baseline_rate, n_per_day, n_days, n_sims, alpha):
    return simulate_peeking_problem(
        baseline_rate= baseline_rate,
        n_per_day_per_group= n_per_day,
        n_days = n_days,
        n_simulation = n_sims,
        alpha= alpha,
        seed=123,
    )

df = load_data()

st.title("A/B Testing & Experimentation Simulator")
st.caption(
    "A check-out page redesign experiment, simulated end-to-end: " \
    "design -> analyze -> compare statistical philosophies."
)

if df is None:
    st.error(
        "No data found at data/experiment_data.csv. Run simulate_data.py first."
    )
    st.stop()

control = df[df["variant"] == "control"]
treatment = df[df["variant"] == "treatment"]

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Experiment Overview",
        "Power Analysis",
        "Results - Frequentists & the Peeking Problem",
        "Result - Bayesian",
    ]
)

# TAB 1
with tab1:
    st.header("What's being tested?")
    st.markdown(
        """
        - **The Experiment:** an e-commerce checkout page redesign.
          Users are randomly split into two groups
        - **Control** - sees the existing checkout page
        - **Treatment** - sees the redesign checkout page

        - **Primary metric:** conversion - did the user complete a purchase? (yes/no)

        - **Goal:** determine whether the redesign genuinely increase the
        conversion rate, and by how much, using proper statistical methods
        (not just eyeballing the numbers).
        """
    )

    col1, col2, col3, col4, = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Control Users", f"{len(control):,}")
    col3.metric("Treatment Users", f"{len(treatment):,}")
    col4.metric("Days Running", f"{df['day'].nunique()}")

    col1, col2, col3 = st.columns(3)
    control_rate = control["conversion"].mean()
    treatment_rate = treatment["conversion"].mean()
    observed_lift = treatment_rate - control_rate
    col1.metric("Control conversion rate", f"{control_rate*100:.2f}%")
    col2.metric("Treatment conversion rate", f"{treatment_rate*100:.2f}%")
    col3.metric("Observed lift", f"{observed_lift*100:+.2f} pts")

    st.subheader("Daily conversion rate over time")
    daily = df.groupby(["day", "variant"])["conversion"].mean().reset_index()
    daily_pivot = daily.pivot(index = "day", columns="variant", values="conversion")
    st.line_chart(daily_pivot)
 
    st.info(
        "💡 **Plain-English summary:** Over the course of the experiment, "
        f"the treatment group converted at {treatment_rate*100:.2f}% versus "
        f"{control_rate*100:.2f}% for control — a difference of "
        f"{observed_lift*100:+.2f} percentage points. Whether that's a "
        "real effect or just noise is exactly what the next two tabs dig into."
    )

# TAB 2
with tab2:
    st.header("Power Analysis: How many users do you need?")
    st.markdown(
        """
        Before running an experiment, you should figure out how many users
        you need to reliably effect you care about.
        Adjust the sliders below to see how the required sample size changes.
        """
    )
    with st.expander("What do these terms mean?"):
        st.markdown(
            """
            - **Baseline conversion rate:** Your current (control) conversion rate.
            - **Minimum Detectable Effect(MDE):** The smallest real improvement you
            actually care about catching. Smaller MDE + harder to detect = needs many more users.
            - **Statistical power:** The probability your test correctly detects
            the effect, if it's real. 80% power means: if the effect is real,
            you'd catch it in about 80 out of 100 repeated experiments.
            - **Significance level(alpha):** Your acceptable false-positive rate.
            Standard is 5%.
            """
        )

    col1, col2, = st.columns(2)
    with col1:
        baseline_input = st.slider("Baseline conversion effect(%)", 1.0,30.0,8.0,0.5)/100
        mde_input = st.slider("Minimum Detectable Effect (% points)", 0.1,10.,1.5,0.1)/100
    with col2:
        power_input = st.slider("Desired power(%)", 50,99,80,1)/100
        alpha_input = st.slider("Significance level/ alpha (%)",1,20,5,1)/100

    n_needed = required_sample_size(baseline_input, mde_input, power= power_input, alpha=alpha_input)
    st.metric("Required sample size PER GROUP", f"{n_needed:,}")
    st.metric("Required TOTAL sample size", f"{n_needed*2:,}")

    st.subheader("Our actual experiment, evaluated")
    actual_n = len(control)
    achieved = achieved_power(baseline_input, mde_input, actual_n, alpha = alpha_input)
    st.write(
           f"Our simulated experiment used **{actual_n:,} users per group**. "
        f"Given the settings above, that gives an achieved power of "
        f"**{achieved*100:.1f}%** "
        + (" (meets or exceeds your target)" if achieved >= power_input
            else " (below your target — a real experiment with this sample size risks missing a true effect)")
    )

    # small visual: sample size needed vs MDE
    st.subheader("Sample size needed vs. effect size you want to detect")
    mde_range = np.linspace(0.005, 0.03, 25)
    n_range = [required_sample_size(baseline_input, m, alpha= alpha_input, power=power_input)for m in mde_range]
    fig,ax = plt.subplots(figsize=(7,3.5))
    ax.plot(mde_range*100, n_range, marker="o", marksize =3)
    ax.set_xlabel("Minimum Detectable Effect (percentage points)")
    ax.set_ylabel("Required sample size per group")
    ax.set_title("Smaller effects require dramatically more users")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# TAB 3
with tab3:
    st.header("Frequentist Results")

    result= two_proportion_ztest(
        control["conversion"].sum(), len(control),
        treatment["conversion"].sum(), len(treatment),
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("p-value", f"{result['p-value']:.4f}")
    col2.metric("Effect (treatment - control)", f"{result['effect']*100:+2f} pts")
    col3.metric("Significant at α=0.05?","Yes" if result["significant"] else "No",)

    st.write(
        f"**95% Confidence Interval on the effect:** [{result['ci_lower']*100:+.2f}, {result['ci_upper']*100:+.2f}] percentage points"
    )
    st.success(result["summary"]) if result["significant"] else st.warning(result["summary"])

    st.divider()
    st.header("The Peeking Problem")
    st.markdown(
        """
        **What happens if you check your p-value every day and stop as soon
        as it's significant, instead of waiting for your planned sample size?**
 
        Below, we simulate hundreds of fake experiments where control and
        treatment have the **exact same true conversion rate** — meaning
        there is **no real effect**, so any "significant" result we find is
        by definition a **false positive**. We compare two strategies:
 
        - **Fixed-horizon**: only look at the p-value once, at the end.
        - **Daily peeking**: check the p-value every day, stop and declare a
          win the first day it crosses 0.05.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.slider("Number of simulated experiments", 200, 2000, 1000, 100)
    with col2:
        n_days_sim = st.slider("Days per simulated experiment", 5, 30, 14, 1)

    with st.spinner("Running peeking problem simulation"):
        peek = run_peeking_simulation(
            baseline_rate = control_rate, n_per_day=350, n_days=n_days_sim, n_sims=n_sims,
            alpha = 0.05)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected false positive rate", "5.0%")
    col2.metric("Fixed-horizon (actual)", f"{peek['fixed_horizon_fpr']*100:.1f}%")
    col3.metric(
        "Daily peeking (actual)",
        f"{peek['peeking_fpr']*100:.1f}%",
        delta=f"{(peek['peeking_fpr']-0.05)*100:+.1f} pts vs expected",
        delta_color="inverse",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Expected\n(alpha)", "Fixed-horizon\n(actual)", "Daily peeking\n(actual)"],
        [0.05, peek["fixed_horizon_fpr"], peek["peeking_fpr"]],
        color=["gray", "#2ca02c", "#d62728"],
    )
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("False-positive rate")
    ax.set_title("Peeking daily inflates your false-positive rate")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height*100:.1f}%", (bar.get_x() + bar.get_width()/2, height),
                     textcoords="offset points", xytext=(0, 5), ha="center")
    st.pyplot(fig)

    inflation = peek["peeking_fpr"] / 0.05
    st.error(
        f"💡 **Key finding:** Checking results daily and stopping early inflates "
        f"the false-positive rate to about **{inflation:.1f}x** the 5% you thought "
        f"you were getting. This is why real experimentation platforms either "
        f"commit to a fixed sample size in advance, or use special statistical "
        f"corrections (like sequential testing) designed for repeated looks — "
        f"which is a related but separate topic from this project's scope."
    )
 
    st.subheader("Example: one simulated experiment's p-value over time")
    example_pvals = peek["daily_pvalues_example"]
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(range(1, len(example_pvals) + 1), example_pvals, marker="o", markersize=4)
    ax2.axhline(0.05, color="red", linestyle="--", label="alpha = 0.05")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("p-value (cumulative)")
    ax2.set_title("A p-value can wander below 0.05 by pure chance, even with no real effect")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# TAB 4
with tab4:
    st.header("Bayesian Results")
    st.markdown(
        "Instead of asking *'how surprising is this data if there's no effect?'*," \
        "the Bayesian approach directly estimates the probability distribution" \
        "of each group's true conversion rate, and lets us make direct" \
        "probability statements like*'there's a 99% chance treatment is better.'*"
    )
    control_post_a, control_post_b = compute_posterior(control['conversion'].sum(), len(control))
    treatment_post_a, treatment_post_b = compute_posterior(treatment['conversion'].sum(), len(treatment))

    control_summary = posterior = posterior_summary(control_post_a, control_post_b)
    treatment_summary = posterior_summary(treatment_post_a, treatment_post_b)
    bayes_result = probability_treatment_beats_control(
        control_post_a, control_post_b, treatment_post_a, treatment_post_b
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("P(treatment > control)", f"{bayes_result['prob_treatment_better']*100:.1f}%")
    col2.metric("Expected uplift", f"{bayes_result['expected_uplift']*100:+.2f} pts")
    col3.metric(
        "95% credible interval",
        f"[{bayes_result['uplift_ci_lower']*100:+.2f}, {bayes_result['uplift_ci_upper']*100:+.2f}]",
    )

    st.subheader("Posterior distributions: control vs treatment conversion rate")
    x = np.linspace(0.04,0.13,500)
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(x, stats.beta.pdf(x, control_post_a, control_post_b), label ="Control",
            color = "#1f77b4")
    ax.fill_between(x, stats.beta.pdf(x, control_post_a, control_post_b), alpha= 0.3, color="#1f77b4")
    
    ax.plot(x, stats.beta.pdf(x, treatment_post_a, treatment_post_b), label ="Treatment",
                color = "#ff7f0e")
    ax.fill_between(x, stats.beta.pdf(x, treatment_post_a, treatment_post_b), alpha= 0.3, color="#ff7f0e")

    ax.set_xlable("True Conversion Rate")
    ax.set_ylabel("Posterior Density")
    ax.set_title("What we believe the true conversion rate is, after seeing the data")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Posterior distribution of the uplift(treatment- control)")
    fig2, ax2 = plt.subplots(figsize=(7,3.5))
    ax2.hist(bayes_result["uplift_samples"]*100, bins=80, color="#2ca02c", alpha=0.7)
    ax2.set_xlabel("Uplift (% points)")
    ax2.set_ylabel("Frequency (from posterior samples)")
    ax2.set_title("Distribution of the plausible true uplift")
    st.pyplot(fig2)

    st.divider()
    st.subheader("Side by Side :Frequentist vs Bayesian")

    freq_result = two_proportion_ztest(
        control["conversion"].sum(), len(control),
        treatment["conversion"].sum(), len(treatment),
    )
    bayes_confident= bayes_result["prob_treatment_better"] > 0.95 or bayes_result["prob_treatment_better"] < 0.05
    agree = freq_result["significant"] == bayes_confident

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Frequentist**")
        st.write(f"p_value = {freq_result['p_value']:.4f}")
        st.write(f"Significant at α=0.05? {'Yes' if freq_result['significant'] else 'No'}")
        st.write(f"Effect estimate: {freq_result['effect']*100:+.2f} pts")
        st.write(f"95% CI: [{freq_result['ci_lower']*100:+.2f}, {freq_result['ci_upper']*100:+.2f}]")
    with col2:
        st.markdown("**Bayesian**")
        st.write(f"P(treatment > control) = {bayes_result['prob_treatment_better']*100:.1f}%")
        st.write(f"Confident conclusion? {'Yes' if bayes_confident else 'No'}")
        st.write(f"Expected uplift: {bayes_result['expected_uplift']*100:+.2f} pts")
        st.write(f"95% credible interval: [{bayes_result['uplift_ci_lower']*100:+.2f}, {bayes_result['uplift_ci_upper']*100:+.2f}]")

        if agree:
            st.success("Both methods agree on the conclusion for this experiment.")
        else:
            st.warning("The two methods land in different places- a good discussionpoint about" \
            "how each framework handles uncertainty.")