"""Entry point for enriching provenance graphs with security features."""

from security_features import SecurityFeatures
from pathlib import Path

if __name__ == "__main__":

    raw_graph = Path("2-Security-features/input/sample_graph.json")
    enriched_graph = Path("2-Security-features/output/graph_with_security_features.json")

    processor = SecurityFeatures(raw_graph, enriched_graph)
    processor.process()
