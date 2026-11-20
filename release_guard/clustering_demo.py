"""Demonstrate requirement 5: treating requests as independent under-states
the variance and inflates false rejections.

Controlled experiment, no engine involved so the effect is not masked by the
exact Bernoulli e-process: each user carries a latent risk factor shared by
all their requests (intra-user correlation).  Two groups are drawn from the
SAME null distribution, and we run two-sample z-tests two ways on identical
data:

* request-level (the anti-pattern): every request is an iid Bernoulli trial,
  standard error uses the request count n_request;
* user-level (correct): the observation is each user's failure fraction,
  standard error uses the number of independent users n_user.

Under the null both should reject at the nominal level (5%); only the
user-level version does.
"""
from __future__ import annotations

import numpy as np


def simulate(n_releases: int = 2000, n_users: int = 600,
             req_per_user: int = 20, base_p: float = 0.02,
             risk_sd: float = 1.0, nominal: float = 0.05) -> dict:
    rng = np.random.default_rng(0)
    z_crit = 1.959963984540054  # two-sided 5%
    reject_request = 0
    reject_user = 0
    var_mean_request = []
    var_mean_user = []
    for _ in range(n_releases):
        # split users into two equal groups under the SAME null
        risk = rng.lognormal(0.0, risk_sd, n_users)
        half = n_users // 2
        rows = {"request": [], "user": []}
        for g, idx in ((0, range(0, half)), (1, range(half, n_users))):
            user_fracs = []
            request_vals = []
            for u in idx:
                p = min(0.5, base_p * risk[u])
                draws = (rng.random(req_per_user) < p).astype(float)
                user_fracs.append(draws.mean())
                request_vals.extend(draws.tolist())
            rows["request"].append(np.array(request_vals))
            rows["user"].append(np.array(user_fracs))
        for level in ("request", "user"):
            a, b = rows[level]
            diff = b.mean() - a.mean()
            se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
            z = diff / max(se, 1e-12)
            if abs(z) > z_crit:
                if level == "request":
                    reject_request += 1
                else:
                    reject_user += 1
        # estimated variance of the overall mean under the two conventions
        all_user = rows["user"][0]
        all_req = rows["request"][0]
        var_mean_user.append(all_user.var(ddof=1) / all_user.size)
        var_mean_request.append(all_req.var(ddof=1) / all_req.size)
    out = {
        "nominal_level": nominal,
        "request_level_false_reject_rate": reject_request / n_releases,
        "user_level_false_reject_rate": reject_user / n_releases,
        "mean_estimated_var_of_mean_request": float(np.mean(var_mean_request)),
        "mean_estimated_var_of_mean_user": float(np.mean(var_mean_user)),
    }
    print(f"零假设两等组，名义水平 {nominal:.0%}，{n_releases} 次模拟：")
    print(f"  请求级（把请求当独立样本）实际拒绝率: "
          f"{out['request_level_false_reject_rate']:.2%}")
    print(f"  用户级（正确的独立抽样单位）实际拒绝率: "
          f"{out['user_level_false_reject_rate']:.2%}")
    print(f"  组均值方差估计  请求级={out['mean_estimated_var_of_mean_request']:.2e}"
          f"  用户级={out['mean_estimated_var_of_mean_user']:.2e}")
    print("请求级样本量虚高（用户的多条请求不独立）→ 标准误被压小 → "
          "零假设下频繁越界，这正是把请求当独立样本的危害。")
    return out


if __name__ == "__main__":
    simulate()
