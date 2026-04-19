import sys
sys.path.insert(0, "src")
from soma_v2.agents.deliberative import _inject_commands
from soma_v2.core.planner import PlanGraph, PlanNode

def test_injection():
    graph = PlanGraph()
    graph.add_node(PlanNode("s1", "Transit to sector 7"))
    graph.add_node(PlanNode("s2", "Relocate A12 to base"))
    graph.add_node(PlanNode("s3", "Check telemetry for B4"))
    graph.add_node(PlanNode("s4", "Inspect zone C3"))
    graph.add_node(PlanNode("s5", "Deactivate node D1"))
    
    # Event text containing unit IDs
    event = "Critical mission involving A12, B4, C3 and D1"
    
    processed = _inject_commands(graph, event)
    
    for node in processed.all_nodes():
        print(f"{node.node_id}: {node.description}")

if __name__ == "__main__":
    test_injection()
