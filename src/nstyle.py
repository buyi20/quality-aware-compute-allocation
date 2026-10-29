"""Plot style, colours and helpers shared by the data figures."""
import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"
LIGHT_BLUE = "#C6DBEF"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)

MM = 1 / 25.4
TEXTWIDTH = 170 * MM          # single-column text width in inches

# colour, label and line style of each processing-cost shape
COST = {"exp": CATEGORICAL[0], "power": CATEGORICAL[2], "log": ACCENT_RED}
COST_LABEL = {"exp": "Exponential", "power": "Power", "log": "Logarithmic"}
LS = {"exp": "-", "power": "--", "log": "-."}
LINE = dict(lw=1.3, alpha=0.9)
MARK = dict(ms=3.8, mew=0)
DOMAIN = {"arxiv": "ArXiv", "book": "Books", "commoncrawl": "Common Crawl", "c4": "C4",
          "stackexchange": "StackExchange", "wikipedia": "Wikipedia", "github": "GitHub"}


def label(ax, letter, x=-0.18, y=1.02):
    """Panel letter in the upper-left corner of an axis."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
