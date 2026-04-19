import graphviz
import os

def generate_soma_architecture():
    dot = graphviz.Digraph('SOMA_Architecture', format='pdf', engine='dot')
    
    # Graph attributes for a clean, academic look
    dot.attr(rankdir='TB', size='10,8!', fontname='Helvetica', nodesep='0.6', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#f8f9fa', 
             color='#343a40', fontname='Helvetica', penwidth='1.5')
    dot.attr('edge', color='#495057', penwidth='1.5', fontname='Helvetica-Oblique', fontsize='10')

    # Input Node
    dot.node('Task', 'Incoming Mission Task\n(e.g., "Scan Sector Alpha")', shape='note', fillcolor='#e9ecef')

    # Tier 1: AgentDirector
    with dot.subgraph(name='cluster_tier1') as c:
        c.attr(label='Tier 1: AgentDirector (Ingestion)', style='rounded', color='#adb5bd', fontname='Helvetica-Bold')
        c.node('Director', 'AgentDirector\n(Task Normalization & Logging)', fillcolor='#d0ebff', color='#1864ab')

    # Tier 2: The SOMA Kernel
    with dot.subgraph(name='cluster_tier2') as c:
        c.attr(label='Tier 2: SOMA V2 Kernel (Orchestration)', style='rounded', color='#adb5bd', fontname='Helvetica-Bold')
        
        c.node('L1', 'L1 Syntax Cache\n(O(1) Hash Lookup)', fillcolor='#fff3bf', color='#e67700')
        c.node('L2', 'L2 Semantic Cache\n(ChromaDB Vector Lookup)', fillcolor='#fff3bf', color='#e67700')
        c.node('Router', 'Hybrid Router\n(DistilBERT Depth Classifier)', fillcolor='#d8f5a2', color='#5c940d')
        
        c.edge('L1', 'L2', label='Miss')
        c.edge('L2', 'Router', label='Miss')

    # Tier 3: Agents
    with dot.subgraph(name='cluster_tier3') as c:
        c.attr(label='Tier 3: Execution Agents', style='rounded', color='#adb5bd', fontname='Helvetica-Bold')
        
        c.node('Reactive', 'Reactive Agent\n(Rule-based / Syntax)', fillcolor='#ffc9c9', color='#c92a2a')
        c.node('Routing', 'Routing Agent\n(Delegation & Splitting)', fillcolor='#ffc9c9', color='#c92a2a')
        c.node('Deliberative', 'Deliberative Agent\n(Local LLM Inference)', fillcolor='#ffc9c9', color='#c92a2a')

    # Tier 4: Resource Broker
    with dot.subgraph(name='cluster_tier4') as c:
        c.attr(label='Tier 4: Contention Resolution', style='dashed', color='#adb5bd', fontname='Helvetica-Bold')
        
        c.node('Broker', 'NegotiationBroker\n(Conflict Resolution)', fillcolor='#eebefa', color='#862e9c')
        c.node('Blackboard', 'ResourceBlackboard\n(Actuator & Telemetry Locks)', fillcolor='#eebefa', color='#862e9c')
        
        c.edge('Broker', 'Blackboard', label='Lock/Claim')

    # Outputs
    dot.node('Output', 'Action Execution\n(Actuator / API Call)', shape='cylinder', fillcolor='#e9ecef')

    # Connections
    dot.edge('Task', 'Director')
    dot.edge('Director', 'L1')
    
    # Cache Hits
    dot.edge('L1', 'Reactive', label='Hit')
    dot.edge('L2', 'Reactive', label='Hit (>0.85 Sim)')
    
    # Router Paths
    dot.edge('Router', 'Reactive', label='Depth 1')
    dot.edge('Router', 'Routing', label='Depth 2')
    dot.edge('Router', 'Deliberative', label='Depth 3')

    # Agents to Broker
    dot.edge('Reactive', 'Broker', label='Resource Request')
    dot.edge('Routing', 'Broker', label='Resource Request')
    dot.edge('Deliberative', 'Broker', label='Resource Request')

    # Broker to Output
    dot.edge('Blackboard', 'Output', label='Granted')

    # Save to paper directory
    output_path = r"c:\Users\HP VICTUS\Documents\my_mcp\urban-swarm-os-v2\paper\soma_v2_architecture"
    dot.render(output_path, cleanup=True)
    print(f"Generated architecture diagram at {output_path}.pdf")

if __name__ == '__main__':
    generate_soma_architecture()
