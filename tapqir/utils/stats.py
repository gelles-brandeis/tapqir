from pyro.ops.stats import pi, quantile


def ci_from_trace(tr, sites, ci=0.95, num_samples=500):
    ci_stats = {}
    for name in sites:
        ci_stats[name] = {}
        hpd = pi(
            tr.nodes[name]["fn"].sample((num_samples,)).data.squeeze().cpu(),
            ci,
            dim=0,
        )
        mean = tr.nodes[name]["fn"].mean.data.squeeze().cpu()
        ci_stats[name]["high"] = hpd[1]
        ci_stats[name]["low"] = hpd[0]
        ci_stats[name]["mean"] = mean
    return ci_stats
