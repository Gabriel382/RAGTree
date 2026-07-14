# ragtree/vendor/byokg/graphstore.py
from abc import ABC, abstractmethod

class GraphStore(ABC):
    """
    Abstract base class for graph store implementations.
    Defines the basic interface for graph operations.
    """
    @abstractmethod
    def get_schema(self):
        """
        Return the graph schema
        :return:
        """
        pass

    @abstractmethod
    def nodes(self):
        """
        Return a list of all node_ids in the graph.

        :return: List[str] all node_ids in the graph
        """
        pass

    @abstractmethod
    def get_nodes(self, node_ids):
        """
        Return node details for given node ids

        :param node_ids: List[str] node ids
        :return: Dict[node_id:Str, Any] node details
        """
        pass

    @abstractmethod
    def edges(self):
        """
        Return a list of all edge_ids in the graph.

        :return: List[str] all edge_ids in the graph
        """
        pass

    @abstractmethod
    def get_edges(self, edge_ids):
        """
        Return edge details for given edge ids

        :param edge_ids: List[str] edge ids
        :return: Dict[edge_id:Str, Any] edge details
        """
        pass

    @abstractmethod
    def find_edges(self, node_ids=None, rel_types=None, direction="out"):
        """
        Find edges given node_ids and relationship types.

        :param node_ids: Optional[List[str]]
        :param rel_types: Optional[List[str]]
        :param direction: "out" | "in" | "both"
        :return: List[str] edge_ids
        """
        pass
