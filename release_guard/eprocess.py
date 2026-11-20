"""E-processes for sequential, always-valid hypothesis testing.

Classical fixed-sample tests (t-test, chi-square ...) assume the sample size is
fixed before data are seen.  Peeking at the data after every batch and stopping
when a naive p-value crosses 0.05 inflates the false-positive rate without bound
as the number of looks grows.

The tools here are *e-processes*: sequences (E_n) that are nonnegative
supermartingales under the null, E_0 = 1.  Ville's inequality gives

    P_0(exists n: E_n >= 1/alpha) <= alpha

for ANY stopping time -- the false-positive promise holds no matter when the
operator stops looking.  Inverting E_n(theta) < 1/alpha over theta yields an
always-valid confidence sequence whose simultaneous coverage is 1-alpha.

Two test families are provided:

* :func:`msprt_two_sided` / :func:`msprt_harm` -- asymptotic two-sample
  mSPRT e-processes for a difference of user-level means (covers proportions,
  counts and threshold-exceedance rates once they are expressed as per-user
  means).  The Gaussian-mixture prior variances are FIXED before the release
  from pre-release baseline dispersion, so the test cannot be destabilised by
  tiny early samples; a grid of mixture scales is averaged (a mixture of
  e-processes is again an e-process), giving robustness to unknown effect size.
* :func:`BernoulliSequential` -- exact Jeffreys-mixture e-process for a
  Bernoulli proportion against a (possibly time-varying) null, used for the
  sample-ratio randomisation check.

Everything is NumPy + the standard library only.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Mixture prior scales relative to the baseline per-user variance s0^2:
# tau^2 = rho * s0^2.  Fixed from PRE-RELEASE data, never touched by the
# sequential observations.  Spans ~5% to ~2x baseline-SD effect sizes.
RHO_GRID: tuple = (0.0025, 0.01, 0.04, 0.16, 0.64, 2.56)

_EPS = 1e-12
_V_FLOOR = 1e-10
_Z_CLIP = 37.0          # exp(-z^2/2) underflows beyond this anyway


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised (Hastings 1955 approximation).

    |abs error| < 7.5e-8, ample for test statistics and posterior tails.
    """
    x = np.clip(np.asarray(x, dtype=float), -_Z_CLIP, _Z_CLIP)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530
                + t * (-0.356563782
                        + t * (1.781477937
                               + t * (-1.821255978 + 1.330274429 * t))))
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - phi * poly
    return np.where(x >= 0.0, cdf, 1.0 - cdf)


def _logmeanexp(log_vals: np.ndarray) -> float:
    """Stable mean of exp(log_vals), with a hard representability cap."""
    mx = float(np.max(log_vals))
    if mx > 650.0:
        return math.exp(650.0)
    return float(np.exp(mx + math.log(np.mean(np.exp(log_vals - mx)))))


# ---------------------------------------------------------------------------
# Two-sample mSPRT on user-level means, fixed mixture variance
# ---------------------------------------------------------------------------

def _tau2_grid(baseline_user_var: float) -> np.ndarray:
    # floor at 1e-8 native units: smaller mixture scales would only detect
    # effects nobody would act on, and guard the divisions below against
    # underflow when a slice baseline variance is essentially zero.
    s0 = max(baseline_user_var, 1e-8)
    return np.array(RHO_GRID, dtype=float) * s0


def _log_components(diff: float, var_diff: float, tau2: np.ndarray):
    """Per-mixture-component log LR pieces.

    With observation ``diff | theta ~ N(theta, V)`` and prior
    ``theta ~ N(0, tau^2)``:
        two-sided log LR = 0.5 log(1+V/tau^2)
                           + 0.5 diff^2 / (V + tau^2)
        one-sided (harm, theta >= 0) divides by Phi(diff / sqrt(V + tau^2)).
    All quantities stay finite because tau^2 is fixed and O(baseline var).
    """
    # With small early samples a Bernoulli slice can have zero observed
    # variance, which would make the LR diverge and the root search expand
    # forever.  A floor of 1e-10 native units keeps every statistic finite
    # while corresponding to effects far below any practical floor.
    v = max(var_diff, _V_FLOOR)
    if not math.isfinite(diff):
        diff = 0.0
    z = diff / math.sqrt(v)
    zc = min(max(z, -_Z_CLIP), _Z_CLIP)
    log_e2 = 0.5 * np.log1p(v / tau2) + 0.5 * zc * zc * v / (v + tau2)
    denom = np.sqrt(np.maximum(v + tau2, _EPS))
    tail = normal_cdf(np.clip(diff / denom, -_Z_CLIP, _Z_CLIP))
    log_eh = log_e2 - np.log(np.maximum(tail, _EPS))
    return log_e2, log_eh


def msprt_two_sided(diff: float, var_diff: float,
                    baseline_user_var: float) -> float:
    """Two-sided mSPRT e-value for H0: treatment difference == 0."""
    log_e2, _ = _log_components(diff, var_diff,
                                _tau2_grid(baseline_user_var))
    return _logmeanexp(log_e2)


def msprt_harm(diff: float, var_diff: float,
               baseline_user_var: float) -> float:
    """One-sided e-value for the harmful direction H1: difference > 0.

    Harm is always coded so a positive difference means the new version is
    worse.  The one-sided mixture uses the positive-half Gaussian prior, whose
    marginal likelihood is the two-sided LR divided by Phi(...).
    """
    _, log_eh = _log_components(diff, var_diff,
                                _tau2_grid(baseline_user_var))
    return _logmeanexp(log_eh)


def msprt_harm_at(diff: float, var_diff: float, baseline_user_var: float,
                  theta: float) -> float:
    """One-sided harm e-value re-centred at candidate effect ``theta``."""
    _, log_eh = _log_components(diff - theta, var_diff,
                                _tau2_grid(baseline_user_var))
    return _logmeanexp(log_eh)


def msprt_two_sided_at(diff: float, var_diff: float, baseline_user_var: float,
                       theta: float) -> float:
    """Two-sided mSPRT e-value against H0: true difference == ``theta``."""
    log_e2, _ = _log_components(diff - theta, var_diff,
                                _tau2_grid(baseline_user_var))
    return _logmeanexp(log_e2)


def confidence_sequence(diff: float, var_diff: float,
                        baseline_user_var: float,
                        alpha: float) -> tuple[float, float]:
    """Two-sided (1-alpha) always-valid confidence sequence.

    For each mixture component the acceptance set of
    ``LR_component(theta) < 1/alpha`` is an interval whose endpoints are the
    closed-form roots of a quadratic in theta; the mixture's acceptance set is
    the intersection across components (a value is rejected only if the
    averaged e-process crosses 1/alpha, so intersecting component sets is
    conservative but exactly valid).  No iterative search is needed.
    """
    v = max(var_diff, _EPS)
    target = 1.0 / alpha
    lowers, uppers = [], []
    for tau2 in _tau2_grid(baseline_user_var):
        # 0.5 log(1+V/tau2) + (diff-theta)^2 / (2(V+tau2)) = log(1/alpha)
        a = v + tau2
        rhs = 2.0 * (math.log(target) - 0.5 * math.log1p(v / tau2)) * a
        disc = diff * diff - rhs
        if disc <= 0.0:
            # this component alone never reaches the threshold: it imposes no
            # finite bound, skip it
            continue
        d = math.sqrt(disc)
        lowers.append(diff - d)
        uppers.append(diff + d)
    if not lowers:
        se = math.sqrt(v)
        return diff - 6.0 * se, diff + 6.0 * se
    return max(lowers), min(uppers)


def harm_upper_bound(diff: float, var_diff: float,
                     baseline_user_var: float, alpha: float) -> float:
    """One-sided (1-alpha) always-valid upper confidence bound on harm.

    Rather than bisecting the one-sided LR (which costs dozens of e-value
    evaluations per cell per look), the one-sided bound is derived from the
    closed-form two-sided confidence sequence.  A two-sided (1-alpha) CS has
    simultaneous coverage 1-alpha, so its upper endpoint is also a valid
    one-sided (1-alpha) upper bound -- conservative by at most the factor
    used to allocate alpha between tails, but exact and branch-free.
    """
    _, upper = confidence_sequence(diff, var_diff, baseline_user_var, alpha)
    return float(upper)


# ---------------------------------------------------------------------------
# Exact two-sample Bernoulli e-process (beta-binomial Jeffreys mixture)
# ---------------------------------------------------------------------------

class TwoSampleBernoulliE:
    """Exact e-process for two Bernoulli groups (Jeffreys beta-binomial).

    Gaussian mSPRT e-values are anti-conservative for sparse Bernoulli
    outcomes (~1% rates) at small per-group n.  This class uses the exact
    construction instead.  Each group's marginal likelihood under a
    Jeffreys Beta(1/2, 1/2) prior is

        M(k, n) = pi^-1 * B(k+1/2, n-k+1/2)

    The two-sided e-process divides the product of the two group marginals
    by the marginal of the POOLED data (the null "one common rate" model):

        E_two = M(k1,n1) M(k0,n0) / M(k1+k0, n1+n0)

    The ratio has null expectation <= 1 (the pooled null marginal is the
    Bayes normalising constant) and remains an e-process at every sample
    size and under optional stopping.  The one-sided harm version divides by
    the directional posterior factor Phi(z), floored at 0.05 so a degenerate
    early tail cannot inflate it beyond the two-sided value.

    The engine feeds CUMULATIVE user-level event counts, so the closed-form
    marginal is simply recomputed at every look -- no incremental factor.
    """

    def __init__(self) -> None:
        self.log_e_two = 0.0
        self.log_e_harm = 0.0
        self.k1 = self.n1 = self.k0 = self.n0 = 0

    @staticmethod
    def _log_marginal(k: int, n: int) -> float:
        if n <= 0:
            return 0.0
        return (math.lgamma(k + 0.5) + math.lgamma(n - k + 0.5)
                - math.lgamma(n + 1.0) - math.log(math.pi))

    def update(self, k1: int, n1: int, k0: int, n0: int) -> None:
        """Recompute the e-process from cumulative group event counts."""
        k1, n1, k0, n0 = int(k1), int(n1), int(k0), int(n0)
        self.k1, self.n1, self.k0, self.n0 = k1, n1, k0, n0
        log_alt = self._log_marginal(k1, n1) + self._log_marginal(k0, n0)
        log_null = self._log_marginal(k1 + k0, n1 + n0)
        self.log_e_two = log_alt - log_null
        if n1 > 1 and n0 > 1:
            p1, p0 = k1 / n1, k0 / n0
            v = max(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0, _V_FLOOR)
            z = min(max((p1 - p0) / math.sqrt(v), -_Z_CLIP), _Z_CLIP)
            corr = math.log(max(float(normal_cdf(np.array([z]))[0]), 0.05))
        else:
            corr = math.log(0.5)
        self.log_e_harm = self.log_e_two - corr

    @property
    def e_two(self) -> float:
        return math.exp(min(self.log_e_two, 700.0))

    @property
    def e_harm(self) -> float:
        return math.exp(min(self.log_e_harm, 700.0))


# ---------------------------------------------------------------------------
# Exact Bernoulli e-process (Jeffreys beta(1/2, 1/2) mixture) -- SRM check
# ---------------------------------------------------------------------------

def bernoulli_jeffreys_loge(successes: float, trials: float,
                            p0: float) -> float:
    """Log e-value for Bernoulli data vs fixed null proportion p0.

    Jeffreys' prior Beta(1/2, 1/2) yields a closed-form marginal likelihood,
    so this is an exact (not asymptotic) test process under any optional
    stopping rule.
    """
    p0 = min(max(p0, _EPS), 1.0 - _EPS)
    s = float(successes)
    n = float(trials)
    if n <= 0.0:
        return 0.0
    a = s + 0.5
    b = n - s + 0.5
    log_beta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(n + 1.0))
    return (log_beta - math.log(math.pi)
            - a * math.log(p0) - b * math.log1p(-p0))


def bernoulli_jeffreys_e(successes: float, trials: float, p0: float) -> float:
    """E-value version of :func:`bernoulli_jeffreys_loge`."""
    return math.exp(bernoulli_jeffreys_loge(successes, trials, p0))


class BernoulliSequential:
    """Exact Jeffreys e-process when the null proportion changes over time.

    The treatment traffic fraction is ramped deliberately, so the sample-ratio
    null p0 differs from batch to batch.  A single fixed-p0 beta-binomial
    marginal would then be invalid.  This class multiplies one-batch
    predictive factors derived from exact per-draw Jeffreys predictive
    probabilities: under the null each factor has conditional expectation 1,
    so the product is a test martingale for any sequence of null proportions
    and any stopping time.
    """

    def __init__(self) -> None:
        self.a = 0.5   # Beta posterior Beta(a, b) parameters
        self.b = 0.5
        self.log_e = 0.0
        self.n = 0
        self.k = 0

    def update(self, successes: int, trials: int, p0: float) -> float:
        """Add one batch; return the cumulative log e-value."""
        s = int(successes)
        m = int(trials)
        if m <= 0:
            return self.log_e
        p0 = min(max(float(p0), _EPS), 1.0 - _EPS)
        # Log predictive mass of s successes then m-s failures under the
        # running Beta(a,b) posterior: products of a/(a+b), (a+1)/(a+b+1)...
        log_pred = (
            math.lgamma(self.a + s) - math.lgamma(self.a)
            + math.lgamma(self.b + m - s) - math.lgamma(self.b)
            - math.lgamma(self.a + self.b + m)
            + math.lgamma(self.a + self.b))
        log_null = s * math.log(p0) + (m - s) * math.log1p(-p0)
        self.log_e += log_pred - log_null
        self.a += s
        self.b += m - s
        self.n += m
        self.k += s
        return self.log_e

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e, 700.0))

    @property
    def p_hat(self) -> float:
        return self.k / self.n if self.n else float("nan")


# ---------------------------------------------------------------------------
# Normal posterior helpers (same mixture family as the mSPRT prior) -- ramp
# ---------------------------------------------------------------------------

def posterior_normal_mixture(diff: float, var_diff: float,
                             baseline_user_var: float
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior means, variances and weights under the mSPRT mixture prior.

    With prior theta ~ N(0, tau^2) and observation diff | theta ~ N(theta, V),
    posterior is N(mu_post, s2_post) with
        s2_post = V tau^2 / (V + tau^2),  mu_post = s2_post/V * diff.
    Equal mixture weights across :data:`RHO_GRID` are used.
    """
    v = max(var_diff, _EPS)
    tau2 = _tau2_grid(baseline_user_var)
    s2 = v * tau2 / (v + tau2)
    mu = s2 / v * diff
    w = np.full_like(tau2, 1.0 / len(tau2))
    return mu, s2, w


def expected_positive_harm(diff: float, var_diff: float,
                           baseline_user_var: float) -> float:
    """Posterior expectation of max(theta, 0), averaged over mixture comps.

    E[max(theta,0)] for theta ~ N(mu, s2) is
        mu Phi(mu/s) + s phi(mu/s).
    """
    mu, s2, w = posterior_normal_mixture(diff, var_diff, baseline_user_var)
    s = np.sqrt(s2)
    z = np.clip(mu / s, -_Z_CLIP, _Z_CLIP)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    total = w * (mu * normal_cdf(z) + s * phi)
    return float(total.sum())
