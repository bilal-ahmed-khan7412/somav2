import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper Functions
def draw_box(ax, x, y, width, height, text, bg_color, text_color="white", fontsize=10, font_weight="bold", alpha=0.9, edge_color="black"):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05,rounding_size=0.1",
                                 linewidth=1.5, edgecolor=edge_color, facecolor=bg_color, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', 
            color=text_color, fontsize=fontsize, fontweight=font_weight, zorder=3, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, text="", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333", 
                                connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # Slight offset for text
        ax.text(mx, my + 0.15, text, ha='center', va='center', fontsize=9, 
                color="#222222", fontweight="bold", zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3))

def setup_canvas(title):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # Background
    bg = patches.FancyBboxPatch((0.5, 0.5), 9, 9, boxstyle="round,pad=0.2",
                                linewidth=1.5, edgecolor="#cccccc", facecolor="#f8f9fa", zorder=0)
    ax.add_patch(bg)
    ax.text(5, 9.5, title, ha='center', va='center', fontsize=14, fontweight="bold", color="#001219")
    return fig, ax

# 1. Agent Director
def generate_director_diagram():
    fig, ax = setup_canvas("AgentDirector & A2A Architecture")
    
    # Outer Director Box
    draw_box(ax, 1.5, 2.5, 7, 6, "", "#e0fbfc", edge_color="#98c1d9")
    ax.text(5, 8.2, "AgentDirector", ha='center', fontsize=12, fontweight="bold", color="#005f73")
    
    # Internal Components
    draw_box(ax, 2, 6.5, 6, 1, "A2A Message Bus\n(Pub/Sub Event Routing)", "#005f73")
    draw_box(ax, 2, 4.5, 2.5, 1.5, "Negotiation Broker\n- Detects Conflicts\n- Proposes Yields", "#0a9396")
    draw_box(ax, 5.5, 4.5, 2.5, 1.5, "Resource Blackboard\n- Atomic Locks\n- State Introspection", "#0a9396")
    
    draw_box(ax, 3.5, 3, 3, 0.8, "Load Balancer & Dispatch", "#94d2bd", text_color="black")
    
    # Arrows
    draw_arrow(ax, 5, 6.5, 5, 5.25, "State Sync")
    draw_arrow(ax, 3.25, 4.5, 3.25, 3.8, "")
    draw_arrow(ax, 6.75, 4.5, 6.75, 3.8, "")
    
    plt.tight_layout()
    plt.savefig("media/arch_1_agent_director.png", dpi=300, bbox_inches="tight")
    plt.savefig("paper/arch_1_agent_director.pdf", dpi=300, bbox_inches="tight")

# 2. V2 Kernel & Classifier
def generate_kernel_diagram():
    fig, ax = setup_canvas("V2 Kernel & Depth Classifier Architecture")
    
    draw_box(ax, 4, 8, 2, 0.8, "Incoming Task\n(Natural Language)", "#3d5a80")
    
    draw_box(ax, 2, 4.5, 6, 2.5, "", "#e0e1dd", edge_color="#778da9")
    ax.text(5, 6.7, "V2 Kernel (Depth Classifier)", ha='center', fontsize=12, fontweight="bold", color="#1b263b")
    
    draw_box(ax, 2.5, 5, 2.2, 1.2, "Feature Extraction\n- Entity Count\n- Action Verbs", "#415a77")
    draw_box(ax, 5.3, 5, 2.2, 1.2, "ML Classifier\n(scikit-learn)", "#415a77")
    
    draw_arrow(ax, 5, 8, 5, 7)
    draw_arrow(ax, 4.7, 5.6, 5.3, 5.6)
    
    # Outputs
    draw_box(ax, 1, 2, 2, 1, "Reactive Agent (D1)\nScore < 0.3\nRule-based", "#2a9d8f")
    draw_box(ax, 4, 2, 2, 1, "Routing Agent (D2)\nScore 0.3 - 0.7\n1 LLM Call", "#e9c46a", text_color="black")
    draw_box(ax, 7, 2, 2, 1, "Deliberative (D3)\nScore > 0.7\nMulti-step", "#e76f51")
    
    draw_arrow(ax, 5.5, 5, 2, 3, "Simple")
    draw_arrow(ax, 6.4, 5, 5, 3, "Medium")
    draw_arrow(ax, 7.0, 5, 8, 3, "Complex")
    
    plt.tight_layout()
    plt.savefig("media/arch_2_v2_kernel.png", dpi=300, bbox_inches="tight")
    plt.savefig("paper/arch_2_v2_kernel.pdf", dpi=300, bbox_inches="tight")

# 3. Deliberative Agent (D3)
def generate_d3_agent_diagram():
    fig, ax = setup_canvas("Deliberative Agent (D3) Semantic Planning")
    
    draw_box(ax, 4, 8, 2, 0.8, "Complex Task", "#333333")
    
    draw_box(ax, 2, 6.5, 6, 1, "Semantic Cache Check (L2)", "#457b9d")
    
    # Hit Path
    draw_arrow(ax, 2, 7, 1, 4, "Cache Hit\n(Zero-Shot)", rad=-0.3)
    draw_box(ax, 0.5, 3.5, 2, 1, "Immediate Execution\n(>16,000x Speedup)", "#2a9d8f")
    
    # Miss Path
    draw_arrow(ax, 5, 6.5, 5, 5.5, "Cache Miss")
    draw_box(ax, 3.5, 4.5, 3, 1, "LLM Planner\n(Multi-step Reasoning)", "#1d3557")
    
    draw_arrow(ax, 5, 4.5, 5, 3.5)
    draw_box(ax, 3.5, 2.5, 3, 1, "DAG Plan Execution\n(Actuator Bridge)", "#e63946")
    
    draw_arrow(ax, 6.5, 5, 8, 7, "Store Plan for Future", rad=0.2)
    
    plt.tight_layout()
    plt.savefig("media/arch_3_deliberative_d3.png", dpi=300, bbox_inches="tight")
    plt.savefig("paper/arch_3_deliberative_d3.pdf", dpi=300, bbox_inches="tight")

# 4. Hierarchical Memory
def generate_memory_diagram():
    fig, ax = setup_canvas("Hierarchical Memory Architecture")
    
    draw_box(ax, 3.5, 8, 3, 0.8, "Agent Read/Write Request", "#000000")
    
    draw_box(ax, 1.5, 5, 3, 2, "L1 HotMemory\n- LRU Cache\n- In-Memory Dict\n- Latency: <0.1ms", "#d62828")
    
    draw_box(ax, 5.5, 5, 3, 2, "L2 ColdMemory\n- ChromaDB Vector Store\n- Sentence-Transformers\n- Semantic Search", "#003049")
    
    draw_arrow(ax, 5, 8, 3, 7, "Primary Access")
    draw_arrow(ax, 4.5, 6.5, 5.5, 6.5, "Evict (TTL)")
    draw_arrow(ax, 5, 8, 7, 7, "Cache Miss Lookup")
    
    # Disk
    draw_box(ax, 4, 2, 2, 1, "Persistent Disk Storage\n(archive_*.txt)", "#780000")
    draw_arrow(ax, 7, 5, 5, 3)
    
    plt.tight_layout()
    plt.savefig("media/arch_4_hierarchical_memory.png", dpi=300, bbox_inches="tight")
    plt.savefig("paper/arch_4_hierarchical_memory.pdf", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    generate_director_diagram()
    generate_kernel_diagram()
    generate_d3_agent_diagram()
    generate_memory_diagram()
    print("Successfully generated all 4 granular architecture diagrams in media/ and paper/ folders!")
