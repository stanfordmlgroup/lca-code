import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import OrderedDict
from operator import itemgetter


class FigurePlotter(object):
    def __init__(self, logger, figsize=12):
        self.logger = logger
        self.figure_dir = logger.figure_dir
        self.figure_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.colors = {"Maj": '#073B4C',
                       "Rad1": '#EF476F',
                       "Rad2": '#06D6A0',
                       "Rad3": '#118AB2',
                       "Upper": '#FF9B85',
                       "Lower": '#FFD97D'}

    def _set_axes(self, ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.01])
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 

    def plot_curve(self, ax, x, y, area):
        ax.plot(x, y, label = f'Model (AUC = {area:0.2f})',
                color='#777777', linewidth = 2)

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(sorted(list(zip(labels, handles)),
                               key=itemgetter(0)))
        ax.legend(by_label.values(), by_label.keys(), loc='best',
                  fontsize = 'x-small', framealpha=0.3, fancybox=False)

    def plot_roc(self, curve, area):
        """Plot ROC curve."""
        roc_path = self.figure_dir / "ROC.pdf"
        
        _ = plt.plot(figsize=(self.figsize, self.figsize))
        ax = plt.gca()
        self._set_axes(ax, xlabel="False Positive Rate",
                       ylabel="True Positive Rate", title="ROC Curve")

        x, y, _ = curve
        self.plot_curve(ax, x, y, area)

        self.logger.log(f"Plotting ROC curve to {roc_path}.")
        plt.savefig(roc_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_pr(self, curve, area):
        """Plot precision recall curve."""
        pr_path = self.figure_dir / "PR.pdf"

        _ = plt.plot(figsize=(self.figsize, self.figsize))
        ax = plt.gca()
        self._set_axes(ax, xlabel="Sensitivity",
                       ylabel="Precision", title="PR Curve")

        y, x, _ = curve
        self.plot_curve(ax, x, y, area)

        self.logger.log(f"Plotting PR curve to {pr_path}.")
        plt.savefig(pr_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_figures(self, roc_curve, auroc, pr_curve, auprc):
        self.plot_roc(roc_curve, auroc)
        self.plot_pr(pr_curve, auprc)
