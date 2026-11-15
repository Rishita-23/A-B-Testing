# A/B Testing and Experiment Simulator

An interactive Streamlit dashboard that simulates a checkout-page redesign A/B test and walks through the full experimentation workflow: designing the experiment (power analysis), analyzing results two ways (Frequentist and Bayesian), and demonstrating the "peeking problem" with real simulation.

This is a portfolio and teaching project. The experiment data is simulated, not from a real product.

## Highlights

- **Power analysis:** interactive sample-size calculator (baseline rate, minimum detectable effect, power, significance level).
- **Frequentist analysis:** two-proportion z-test with p-values and confidence intervals.
- **Bayesian analysis:** Beta-Binomial posteriors, probability that treatment beats control, and expected uplift with a credible interval.
- **Peeking problem:** across 1,000 simulated no-effect experiments, checking results daily and stopping at the first significant p-value inflated the false-positive rate about 3x above the stated significance level.

## The simulated experiment

- **Control:** sees the existing checkout page.
- **Treatment:** sees the redesigned checkout page.
- **Metric:** conversion (purchase completed, yes or no).
- **Data:** 5,000 users per group over 14 days, with a known true effect (8% baseline versus 9.5% treatment, a 1.5-point lift), so you can see how well each method recovers the truth.

## Dashboard tabs

1. **Experiment Overview:** users per group, days run, observed conversion rates and lift, and a daily conversion time series.
2. **Power Analysis:** sliders for baseline rate, MDE, power, and alpha, showing required sample size versus the simulated experiment.
3. **Frequentist and Peeking:** the z-test plus a simulation of how daily peeking inflates false positives.
4. **Bayesian:** posterior distributions, probability treatment beats control, expected uplift, and a side-by-side comparison with the Frequentist result.

## Results

| Finding | Value |
| --- | --- |
| True lift built into the data | 8% to 9.5% (1.5 points) |
| Simulated users | 5,000 per group, 14 days |
| False-positive inflation from daily peeking | about 3x (1,000 simulated experiments) |

Add your own observed numbers here after running the app, for example the observed lift, the p-value, and the Bayesian probability that treatment beats control.

## Tech stack

- **Simulation and data:** numpy, pandas
- **Statistics:** scipy.stats (two-proportion z-test, Beta-Binomial model, Monte Carlo posterior sampling)
- **App and visualization:** Streamlit, matplotlib

## Project structure

```
.
├── simulate_data.py        Generates the simulated experiment dataset
├── power_analysis.py       Sample size and power calculations
├── frequentist_testing.py  Z-test and peeking-problem simulation
├── bayesian_testing.py     Beta-Binomial posterior and comparison
├── dashboard.py            Streamlit app
├── data/experiment_data.csv  (generated)
└── requirements.txt
```

## Setup

```
git clone https://github.com/Rishita-23/A-B-Testing.git
cd A-B-Testing
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python simulate_data.py
streamlit run dashboard.py
```

`frequentist_testing.py` and `bayesian_testing.py` also run standalone for a console version of the analysis. Edit `N_PER_VARIANT`, `BASELINE_CONVERSION_RATE`, `TRUE_TREATMENT_EFFECT`, and `N_DAYS` in `simulate_data.py` to try other scenarios.

## Key concepts covered

- **Power analysis:** how many users you need before launching, and the trade-off between MDE and sample size.
- **Two-proportion z-test:** the standard Frequentist comparison of two conversion rates.
- **The peeking problem:** why stopping as soon as p < 0.05 inflates the real false-positive rate.
- **Bayesian A/B testing:** direct probability statements such as "there is a 96% chance treatment beats control".
- **Frequentist versus Bayesian:** the two frameworks can reach different practical conclusions on the same data.

## Limitations

- The data is fully synthetic, so the project demonstrates methodology, not a real product result.
- The peeking simulation reruns every experiment when you move the sliders, so very high simulation counts take a few seconds.
