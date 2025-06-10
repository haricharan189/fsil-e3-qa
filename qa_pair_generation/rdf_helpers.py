from rdflib import Graph, URIRef
from collections import defaultdict
from urllib.parse import unquote
from typing import Dict, Set

class RDFQueryHelper:
    def __init__(self, graph: Graph):
        self.graph = graph


    
    