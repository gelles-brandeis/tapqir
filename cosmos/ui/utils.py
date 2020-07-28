import torch
import matplotlib.pyplot as plt
from pyro.ops.stats import hpdi

def plot_dist(ax, x, trace, params, ci=None, label=None):
    for i, p in enumerate(params):
        hpd = hpdi(trace.nodes[p]["fn"].sample((500,)).data.cpu(), ci, dim=0)
        ax[i].clear()
        if p.endswith("background"):
            ax[i].fill_between(
                x, hpd[0][0], hpd[1][0],
                color="C0", alpha=0.2
            )
            ax[i].scatter(
                x, trace.nodes[p]["fn"].mean.data.cpu(),
                color="C0", s=5, label=label)
        else:
            for k in range(2):
                ax[i].fill_between(
                    x, hpd[0][k, 0], hpd[1][k, 0],
                    color="C{}".format(k), alpha=0.2
                )
                ax[i].scatter(
                    x, trace.nodes[p]["fn"].mean.data[k, 0].cpu(),
                    color="C{}".format(k), s=5, label=label)

        ax[i].set_ylabel(p, fontsize=10)

    ax[len(params)-1].set_xlim(x[0]-2, x[-1]+2)
    ax[len(params)-1].set_xlabel("frame #", fontsize=10)
    plt.tight_layout()
