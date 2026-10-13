# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# line panels      -> LineTrend/plot_sweep.py          -> param inherit (marker lines, alpha 0.8, no grid)
# heatmap panels   -> heatmap/plot_composition.py      -> param inherit (white cell borders, annotated, single-hue)
# strip/violin     -> Violin/plot_Violin.py            -> param inherit (points over light violin)
# parity/scatter   -> basic-plots (cross-type)         -> param inherit
# bars / tornado   -> BarComposition (cross-type Forest)-> param inherit
# RULE: production scripts expect different data structures; panels below are drawn natively with inherited parameters.
import matplotlib
matplotlib.use("Agg")

# Plot style shared by all data figures (Times New Roman lettering)
import matplotlib as mpl
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
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline -- COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

WORK = Path(__file__).resolve().parents[1] / "plot_data"   # result files copied from the analysis pipeline
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
TEXTWIDTH = 170 * MM          # Springer single-column text width
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
LIGHT_BLUE = "#C6DBEF"        # light tint of the primary blue for background fills
# semantic roles, fixed across all figures
COST = {"exp": CATEGORICAL[0], "power": CATEGORICAL[2], "log": ACCENT_RED}   # log cost carries the jump result
COST_LABEL = {"exp": "Exponential", "power": "Power", "log": "Logarithmic"}
LS = {"exp": "-", "power": "--", "log": "-."}
LINE = dict(lw=1.3, alpha=0.9)              # LineTrend: linewidth 3 / markersize 8 at 4.5 in panels, scaled to print
MARK = dict(ms=3.8, mew=0)
DOMAIN = {"arxiv": "ArXiv", "book": "Books", "commoncrawl": "Common Crawl", "c4": "C4",
          "stackexchange": "StackExchange", "wikipedia": "Wikipedia", "github": "GitHub"}


def label(ax, letter, x=-0.18, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    save_cns_figure(fig, str(OUT / name))
    plt.close(fig)
    print("saved", name)
