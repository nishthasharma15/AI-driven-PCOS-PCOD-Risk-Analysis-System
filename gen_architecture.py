import matplotlib.pyplot as plt

# High DPI for poster quality
plt.rcParams["figure.dpi"] = 300

fig, ax = plt.subplots(figsize=(8, 11))
ax.axis("off")

# -------------------------------
# Box style helper
# -------------------------------
def draw_box(text, xy, color):
    ax.text(
        xy[0], xy[1], text,
        ha="center", va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=color,
            edgecolor="#2f2f2f",
            linewidth=1.2
        )
    )

# -------------------------------
# Arrow helper
# -------------------------------
def draw_arrow(start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.6,
            color="#333333"
        )
    )

# -------------------------------
# Box positions
# -------------------------------
boxes = {
    "User Input\n(Streamlit UI)": ((0.5, 0.92), "#b3e5fc"),
    "Data Preprocessing": ((0.5, 0.84), "#80cbc4"),
    "Basic Model\n(No Lab Data)": ((0.25, 0.68), "#c8e6c9"),
    "Clinical Model\n(With Lab Data)": ((0.75, 0.68), "#d1c4e9"),
    "Risk Scoring": ((0.5, 0.54), "#ffe082"),
    "Risk Classification\n(Low / Moderate / High)": ((0.5, 0.44), "#ffccbc"),
    "Recommendations": ((0.5, 0.34), "#f8bbd0"),
    "Output\n(Streamlit UI)": ((0.5, 0.24), "#b3e5fc"),
}

# -------------------------------
# Draw boxes
# -------------------------------
for text, (pos, color) in boxes.items():
    draw_box(text, pos, color)

# -------------------------------
# Draw arrows
# -------------------------------
draw_arrow((0.5, 0.90), (0.5, 0.86))
draw_arrow((0.5, 0.82), (0.25, 0.70))
draw_arrow((0.5, 0.82), (0.75, 0.70))
draw_arrow((0.25, 0.66), (0.5, 0.56))
draw_arrow((0.75, 0.66), (0.5, 0.56))
draw_arrow((0.5, 0.52), (0.5, 0.46))
draw_arrow((0.5, 0.42), (0.5, 0.36))
draw_arrow((0.5, 0.32), (0.5, 0.26))

# -------------------------------
# Save figure
# -------------------------------
plt.savefig("pcos_system_architecture_poster.png", bbox_inches="tight")
plt.show()
