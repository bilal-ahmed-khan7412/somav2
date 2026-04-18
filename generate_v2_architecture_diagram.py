import os

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SOMA V2 Architecture Diagrams</title>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: true, theme: 'dark' });
    </script>
    <style>
        body { 
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #0f172a; 
            color: #e2e8f0; 
            margin: 0; 
            padding: 40px; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
        }
        h1 { color: #38bdf8; }
        h2 { color: #f8fafc; }
        .diagram-container { 
            background-color: #1e293b; 
            padding: 30px; 
            border-radius: 12px; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            margin-bottom: 40px;
            width: 90%;
            max-width: 1200px;
            text-align: center;
        }
        .desc {
            color: #94a3b8;
            margin-bottom: 20px;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <h1>Urban Swarm OS V2 Architecture Visualizer</h1>
    <p>A memory-augmented, negotiation-aware operating system for autonomous drone swarms.</p>

    <div class="diagram-container">
        <h2>1. V2 System Architecture Overview</h2>
        <p class="desc">High-level view of the AgentDirector, Tiered Agents, and Hierarchical Memory.</p>
        <div class="mermaid">
        flowchart TD
            subgraph Director [AgentDirector]
                A2A[A2A Bus]
                BB[Resource Blackboard]
                NB[Negotiation Broker]
            end

            subgraph Dispatch [Load Balancing & Classification]
                Kernel[V2Kernel]
                DC[DepthClassifier\nSimple / Medium / Complex]
            end

            subgraph AgentPool [Tiered Agent Slots]
                direction TB
                D1["ReactiveAgent (D1)\nRule-based, 0 LLM calls"]
                D2["RoutingAgent (D2)\nSingle LLM call"]
                D3["DeliberativeAgent (D3)\nMulti-step planner + cache"]
            end

            subgraph Memory [Hierarchical Memory]
                L1["HotMemory (L1)\nLRU, <0.1ms"]
                L2["ColdMemory (L2)\nChromaDB semantic search"]
            end

            Director ==>|Dispatch| Dispatch
            Dispatch -->|Classify & Route| AgentPool
            
            AgentPool -->|Cache Read/Write| Memory
            L1 -.->|Evict| L2
            L2 -.->|Semantic Retrieve| L1

            classDef dir fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff;
            classDef disp fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#fff;
            classDef agent fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;
            classDef mem fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff;

            class Director,A2A,BB,NB dir;
            class Dispatch,Kernel,DC disp;
            class AgentPool,D1,D2,D3 agent;
            class Memory,L1,L2 mem;
        </div>
    </div>

    <div class="diagram-container">
        <h2>2. Dynamic Routing & Semantic Caching (D3)</h2>
        <p class="desc">Shows how the Depth Classifier routes tasks and how Deliberative Agents use cross-domain semantic caching.</p>
        <div class="mermaid">
        sequenceDiagram
            participant Task as Incoming Task
            participant DC as Depth Classifier
            participant D3 as DeliberativeAgent
            participant Mem as HierarchicalMemory
            participant LLM as LLM Backend

            Task->>DC: Analyze task complexity
            DC-->>D3: Route to D3 (Complex Task)
            
            D3->>Mem: Query Semantic Cache (L2 ChromaDB)
            
            alt Cache Hit (Zero-Shot Transfer)
                Mem-->>D3: Found similar semantic plan
                D3->>Task: Execute Plan (>16,000x speedup)
            else Cache Miss
                Mem-->>D3: No matching plan
                D3->>LLM: Multi-step reasoning
                LLM-->>D3: Generated Plan
                D3->>Mem: Store plan in Cache
                D3->>Task: Execute Plan
            end
        </div>
    </div>

</body>
</html>
"""

def generate():
    output_path = os.path.join(os.getcwd(), "generate_v2_architecture.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Success! Generated V2 architecture visualization file at:\n{output_path}")
    print("Double-click this HTML file to view the generated architecture diagrams in your web browser.")

if __name__ == "__main__":
    generate()
