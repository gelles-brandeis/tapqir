import torch
import matplotlib.pyplot as plt
from pyro.ops.stats import hpdi

def plot_dist(ax, _line, _fill_line, predictions, n, x, trace, params, ci=None, label=None):
    for i, p in enumerate(params):
        if p == "z_probs":
            if _line[p] is None:
                _line[p], = ax[i].plot(
                    x,
                    predictions["z_prob"][n],
                    marker="o", ms=3, color="C2", label="z"
                )
            else:
                _line[p].set_ydata(predictions["z_prob"][n])

            ax[i].set_ylim(-0.02, 1.02)
        #hpd = hpdi(trace.nodes[p]["fn"].sample((500,)).data.cpu(), ci, dim=0)
        elif p.endswith("background"):
            std = trace.nodes[p]["fn"].variance.data.cpu().sqrt()
            mean = trace.nodes[p]["fn"].mean.data.cpu()
            if _line[p] is None:
                _line[p],  = ax[i].plot(
                        x, mean[0], marker="o", color="C0",
                        ms=3, ls=None, label=label
                )
            else:
                _line[p].set_ydata(mean[0])
                _fill_line[p].remove()

            _fill_line[p] = ax[i].fill_between(
                    #x, hpd[0][0], hpd[1][0],
                    x, mean[0] - 2 * std[0], mean[0] + 2 * std[0],
                    color="C0", alpha=0.2
            )

            ax[i].set_ylim(0, mean.max() + 2 * std.max())
        else:
            std = trace.nodes[p]["fn"].variance.data.cpu().sqrt()
            mean = trace.nodes[p]["fn"].mean.data.cpu()
            for k in range(2):
                if _line["{}_{}".format(p, k)] is None:
                    _line["{}_{}".format(p, k)], = ax[i].plot(
                        x, mean[k, 0], marker="o", ms=3, ls=None,
                        color="C{}".format(k), label=label
                    )
                else:
                    _line["{}_{}".format(p, k)].set_ydata(mean[k, 0])
                    _fill_line["{}_{}".format(p, k)].remove()

                _fill_line["{}_{}".format(p, k)] = ax[i].fill_between(
                        #x, hpd[0][k, 0], hpd[1][k, 0],
                        x, mean[k, 0] - 2 * std[k, 0], mean[k, 0] + 2 * std[k, 0],
                        color="C{}".format(k), alpha=0.2
                )

            #if p.endswith("x") or p.endswith("y"):
            #    ax[i].set_ylim(-(14+1)/2, (14+1)/2)
            if p.endswith("height"):
                ax[i].set_ylim(0, mean.max() + 2 * std.max())

        ax[i].set_ylabel(p, fontsize=10)

    ax[len(params)-1].set_xlim(x[0]-2, x[-1]+2)
    ax[len(params)-1].set_xlabel("frame #", fontsize=10)
    plt.tight_layout()
