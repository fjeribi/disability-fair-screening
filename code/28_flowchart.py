"""Participant flow diagram (PHQ-8 analyses; GAD-7 sample counts are given in Methods) in the style of Riedel et al. (2020, Data Science Journal, Figures 1-3):
one main path per sample, each exclusion step in a side box "Exclusion criteria: <reason> (N = x)".
Counts are the sequential exclusions reported in Methods."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FILL, EDGE, INK, MUTED = "#dde4f0", "#1f2a36", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.axis("off"); ax.set_xlim(-1, 105); ax.set_ylim(21, 77)
FS, WM, WS, H = 5.9, 17, 17, 7.4


def box(cx, cy, text, w=WM, h=H):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.2,rounding_size=0.9",
                                fc=FILL, ec=EDGE, lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK, linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=7))


def lane(x, steps):
    """steps: list of ('box', text) / ('excl', text); main boxes on x, exclusions in side boxes to the right"""
    sx = x + WM / 2 + WS / 2 + 1.2
    y = 72
    prev_bottom = None
    for kind, text in steps:
        if kind == "box":
            if prev_bottom is not None:
                arrow(x, prev_bottom, x, y + H / 2 + 0.2)
            box(x, y, text); prev_bottom = y - H / 2 - 0.2; y -= 9.2
        else:
            ax.plot([x, x], [prev_bottom, y], color=EDGE, lw=0.9)
            arrow(x, y, sx - WS / 2 - 0.2, y); box(sx, y, "Exclusion criteria:\n" + text, w=WS)
            prev_bottom = y; y -= 9.2


lane(9, [("box", "NHIS 2019 + 2022\n(N = 59,648)"),
         ("excl", "missing Washington\nGroup answer (N = 95)"),
         ("excl", "no PHQ-8 result or\ndepression indicator\n(N = 1,666)"),
         ("box", "Audit sample\n(N = 57,887)"),
         ("excl", "one PHQ-8 item\nmissing (N = 287)"),
         ("box", "DAFS-I development\n(N = 57,600)")])
lane(43, [("box", "NHIS 2025\n(N = 24,215)"),
          ("excl", "missing Washington\nGroup answer (N = 34)"),
          ("excl", "no PHQ-8 result or\ndepression indicator\n(N = 725)"),
          ("box", "Complete records\n(N = 23,456)"),
          ("excl", "one PHQ-8 item\nmissing (N = 124)"),
          ("box", "DAFS-I test\n(N = 23,332)")])
lane(77, [("box", "NHANES 2021–2023\n(N = 11,933)"),
          ("excl", "age < 18 years\n(N = 3,780)"),
          ("excl", "no depression\nscreener (N = 1,816)"),
          ("excl", "PHQ-8 incomplete\n(N = 841)"),
          ("excl", "missing Washington\nGroup answer (N = 28)"),
          ("box", "Replication sample\n(N = 5,468)")])

fig.savefig("out/fig_flow.png"); print("written out/fig_flow.png")
