import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, bg_color, text_color="white", fontsize=11, font_weight="bold", alpha=0.9, edge_color="black"):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05,rounding_size=0.05",
                                 linewidth=1.5, edgecolor=edge_color, facecolor=bg_color, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', 
            color=text_color, fontsize=fontsize, fontweight=font_weight, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, text="", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333", 
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.1, text, ha='center', va='center', fontsize=9, 
                color="#444444", fontweight="bold", zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

def generate_academic_architecture():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Colors (Academic Theme)
    c_bg_main = "#f8f9fa"
    c_director = "#005f73"
    c_agent = "#0a9396"
    c_memory = "#94d2bd"
    c_kernel = "#e9d8a6"
    c_text_dark = "#001219"

    # Background Panel
    bg_panel = patches.FancyBboxPatch((0.5, 0.5), 9, 9, boxstyle="round,pad=0.2",
                                      linewidth=1.5, edgecolor="#cccccc", facecolor=c_bg_main, zorder=0)
    ax.add_patch(bg_panel)
    ax.text(5, 9.6, "SOMA V2: Urban Swarm OS Architecture", ha='center', va='center', 
            fontsize=16, fontweight="bold", color=c_text_dark)

    # 1. Agent Director (Top)
    draw_box(ax, 1, 7.5, 8, 1.5, "", "white", edge_color="#666666", alpha=1.0)
    ax.text(5, 8.7, "Agent Director", ha='center', va='center', fontsize=12, fontweight="bold", color=c_director)
    draw_box(ax, 1.5, 7.7, 2, 0.8, "A2A Bus", c_director)
    draw_box(ax, 4.0, 7.7, 2, 0.8, "Blackboard", c_director)
    draw_box(ax, 6.5, 7.7, 2, 0.8, "Negotiator", c_director)

    # 2. V2 Kernel & Depth Classifier (Middle)
    draw_box(ax, 3.5, 5.5, 3, 1.0, "V2 Kernel\n(Depth Classifier)", c_kernel, text_color=c_text_dark)

    # 3. Agent Slots / Tiers (Bottom Left/Center)
    draw_box(ax, 1, 3.0, 4.5, 1.5, "", "white", edge_color="#666666", alpha=1.0)
    ax.text(3.25, 4.2, "Tiered Agent Slots", ha='center', va='center', fontsize=11, fontweight="bold", color=c_agent)
    draw_box(ax, 1.2, 3.2, 1.2, 0.8, "D1 (Reactive)\n0 LLM", c_agent, fontsize=9)
    draw_box(ax, 2.6, 3.2, 1.2, 0.8, "D2 (Routing)\n1 LLM", c_agent, fontsize=9)
    draw_box(ax, 4.0, 3.2, 1.3, 0.8, "D3 (Deliberative)\nMulti-step", c_agent, fontsize=9)

    # 4. Hierarchical Memory (Bottom Right)
    draw_box(ax, 6.5, 3.0, 2.5, 1.5, "", "white", edge_color="#666666", alpha=1.0)
    ax.text(7.75, 4.2, "Hierarchical Memory", ha='center', va='center', fontsize=11, fontweight="bold", color="#2a9d8f")
    draw_box(ax, 6.7, 3.6, 2.1, 0.4, "L1: HotMemory (<0.1ms)", c_memory, text_color=c_text_dark, fontsize=9)
    draw_box(ax, 6.7, 3.1, 2.1, 0.4, "L2: Cold (ChromaDB)", c_memory, text_color=c_text_dark, fontsize=9)

    # Connections
    # Director to Kernel
    draw_arrow(ax, 5, 7.5, 5, 6.5, "Task Dispatch")
    # Kernel to Agents
    draw_arrow(ax, 4.5, 5.5, 3.25, 4.5, "Classify & Assign")
    # Agents to Memory
    draw_arrow(ax, 5.5, 3.75, 6.5, 3.75, "Semantic Lookup")
    # Memory back to Agents (Cache Hit)
    ax.annotate("", xy=(5.5, 3.35), xytext=(6.5, 3.35),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333", linestyle="--"))
    ax.text(6.0, 3.15, "Cache Hit\n(16,000x)", ha='center', va='center', fontsize=8, color="#555555", fontweight="bold")

    plt.tight_layout()
    plt.savefig("soma_v2_academic_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig("soma_v2_academic_architecture.pdf", dpi=300, bbox_inches="tight")
    print("Successfully generated soma_v2_academic_architecture.png and .pdf")

if __name__ == "__main__":
    generate_academic_architecture()
