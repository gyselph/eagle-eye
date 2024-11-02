import json
from pathlib import Path
from networkx.readwrite import json_graph
import ipaddress
from networkx import DiGraph

class SecurityFeatures:
    """Add security feature to a provenance graph.

    For this enrichment step, raw features from the input graph are analyzed, and used to infer security features.
    Security features are either in numerical, boolean, or categorical format."""

    def __init__(self, raw_graph: Path, enriched_graph: Path):
        self.raw_graph = raw_graph
        self.enriched_graph = enriched_graph
        self.file_disposition_vals = ["1", "2", "3", "5"]
        self.internet_registry_keys = [
            "\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\UNCAsIntranet".lower(),
            "\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\ProxyBypass".lower(),
            "\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\IntranetName".lower(),
            "\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\AutoDetect".lower()
        ]

    def _add_security_feature_command_line_length(self, graph: DiGraph) -> None:
        """Compute the command line length for process events.
         
        Add a corresponding security feature to all events in the graph."""
        # search for all process events
        process_events = [(node_id, attrs) for (node_id, attrs) in graph.nodes(data=True) if ("event_type" in attrs and attrs["event_type"]=="si_create_process")]
        # for all process events: compute length of command-line string and add value as numerical security feature
        for pe in process_events:
            command_line = pe[1]["cmdline"]
            command_line_numerical = len(command_line)
            graph.nodes[pe[0]]["encoding_length_command_line"] = command_line_numerical

    def _add_security_feature_file_disposition(self, graph: DiGraph) -> None:
        """Compute the security feature `encoding_create_file_disposition` for all file events.
         
        Add a corresponding security feature to all events in the graph."""
        # search for all file events
        file_events = [(node_id, attrs) for (node_id, attrs) in graph.nodes(data=True) if ("event_type" in attrs and attrs["event_type"]=="si_create_file")]
        # for all file events: convert "file disposition" to categorical value with limited number of categories (5)
        for fe in file_events:
            file_disposition_raw = fe[1]["create_file_disposition"]
            file_disposition_categorical = file_disposition_raw if file_disposition_raw in self.file_disposition_vals else "OTHER"
            graph.nodes[fe[0]]["encoding_create_file_disposition"] = file_disposition_categorical

    def _add_security_feature_internet_registry_key(self, graph: DiGraph) -> None:
        """For each registry event, check if the key is internet related.
         
        Add a corresponding security feature to all events in the graph."""
        # search for all windows registry events
        registry_events = [(node_id, attrs) for (node_id, attrs) in graph.nodes(data=True) if ("event_type" in attrs and attrs["event_type"]=="si_set_value_key")]
        # for all registry events: check if registry key is related to internet settings, and add a corresponding security feature
        for re in registry_events:
            registry_key = re[1]["object_name"].lower()
            matches = [k in registry_key for k in self.internet_registry_keys]
            is_internet_key = sum(matches) > 0
            graph.nodes[re[0]]["encoding_internet"] = is_internet_key

    def _add_security_feature_socket_internal_source(self, graph: DiGraph) -> None:
        """For each socket event, check if the source is in the internal network.
         
        Add a corresponding security feature to all events in the graph."""
        # search for all network events
        network_events = [(node_id, attrs) for (node_id, attrs) in graph.nodes(data=True) if ("event_type" in attrs and attrs["event_type"]=="etw_tcp_send_ipv4")]
        # for all registry events: check if socket source is from the internal network, and add a corresponding security feature
        for ne in network_events:
            ip_source = ne[1]["saddr"]
            is_internal_source = ipaddress.ip_address(ip_source).is_private
            graph.nodes[ne[0]]["encoding_ip_source_internal"] = is_internal_source

    def process(self):
        """Add security features to a provenance graph and persist the result."""
        print(f"Computing security features for 1 graph: {self.raw_graph}")
        # load graph
        with open(file = self.raw_graph, mode = "r", encoding = "UTF-8") as f:
            json_data = json.load(f)
        graph = json_graph.tree_graph(json_data)
        # add all security features
        self._add_security_feature_command_line_length(graph)
        self._add_security_feature_file_disposition(graph)
        self._add_security_feature_internet_registry_key(graph)
        self._add_security_feature_socket_internal_source(graph)
        # TODO: add more security features according to your dataset
        # persist result
        head_node = [node_id for node_id, degree in graph.in_degree() if degree == 0][0]
        json_data = json_graph.tree_data(graph, root=head_node)
        self.enriched_graph.parent.mkdir(parents=True, exist_ok=True)
        with self.enriched_graph.open("w") as f:
            json.dump(json_data, f, indent=4)
        print(f"Graph with security features is generated: {self.enriched_graph}")
