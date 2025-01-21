from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import urllib.parse
import os

class KnowledgeGraphBuilder:
    def __init__(self):
        # Define namespaces for ontology classes
        self.org_role = Namespace("http://example.org/org_role/")
        self.org_sub_role = Namespace("http://example.org/org_sub_role/")
        self.person_position = Namespace("http://example.org/person_position/")
        self.location_type = Namespace("http://example.org/location_type/")
        
        # Define namespaces for data instances
        self.person_name = Namespace("http://example.org/person_name/")
        self.org_name = Namespace("http://example.org/org_name/")
        self.loc = Namespace("http://example.org/location/")
        self.rel = Namespace("http://example.org/relation/")
        
        # Base classes namespace
        self.base = Namespace("http://example.org/base/")
        
        # Add these relation types explicitly
        self.g = Graph()
        self.g.add((self.rel.hasPosition, RDF.type, RDF.Property))
        self.g.add((self.rel.isEmployedBy, RDF.type, RDF.Property))
        self.g.add((self.rel.hasEmployee, RDF.type, RDF.Property))
        self.g.add((self.rel.hasLocationAt, RDF.type, RDF.Property))
        self.g.add((self.rel.isLocationOf, RDF.type, RDF.Property))
    
    def extract_ontology_classes(self, data: List[Dict]) -> Graph:
        """
        Extract ontology classes from annotations and create TTL file
        """
        positions: Set[str] = set()
        org_roles: Set[str] = set()
        org_sub_roles: Dict[str, str] = {}  # sub_role -> parent_role mapping
        loc_types: Set[str] = set()
        
        # Extract all classes from annotations
        for doc in data:
            for annotation in doc["annotations"]:
                for result in annotation.get('result', []):
                    if 'value' in result:
                        value = result['value']
                        label = value["hypertextlabels"][0]
                        text = value["text"]
                    
                        if label == "Person Position":
                            positions.add(text)
                        elif label == "Organization Role":
                            org_roles.add(text)
                        elif label == "Organization Sub-Role":
                            org_sub_roles[text] = None  # Temporarily store sub-role
                        elif label == "Location Type":
                            loc_types.add(text)
        
        # Create ontology graph
        g = Graph()
        
        # Add namespace prefixes
        g.bind("org_role", self.org_role)
        g.bind("org_sub_role", self.org_sub_role)
        g.bind("person_position", self.person_position)
        g.bind("location_type", self.location_type)
        g.bind("base", self.base)
        
        # Define base classes
        g.add((self.base.Person, RDF.type, RDFS.Class))
        g.add((self.base.Organization, RDF.type, RDFS.Class))
        g.add((self.base.Location, RDF.type, RDFS.Class))
        
        # Add position subclasses
        for position in positions:
            position_uri = self.person_position[self._clean_uri(position)]
            g.add((position_uri, RDF.type, RDFS.Class))
            g.add((position_uri, RDFS.subClassOf, self.base.Person))
        
        # Add organization role subclasses
        for role in org_roles:
            role_uri = self.org_role[self._clean_uri(role)]
            g.add((role_uri, RDF.type, RDFS.Class))
            g.add((role_uri, RDFS.subClassOf, self.base.Organization))
        
        # Add organization sub-role subclasses
        for sub_role, parent_role in org_sub_roles.items():
            sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
            g.add((sub_role_uri, RDF.type, RDFS.Class))
            
            if parent_role:
                parent_role_uri = self.org_role[self._clean_uri(parent_role)]
                g.add((sub_role_uri, RDFS.subClassOf, parent_role_uri))
            else:
                g.add((sub_role_uri, RDFS.subClassOf, self.base.Organization))
        
        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.location_type[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.base.Location))
        
        return g
    
from rdflib import URIRef

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import urllib.parse
import os

class KnowledgeGraphBuilder:
    def __init__(self):
        # Define namespaces for ontology classes
        self.org_role = Namespace("http://example.org/org_role/")
        self.org_sub_role = Namespace("http://example.org/org_sub_role/")
        self.person_position = Namespace("http://example.org/person_position/")
        self.location_type = Namespace("http://example.org/location_type/")
        
        # Define namespaces for data instances
        self.person_name = Namespace("http://example.org/person_name/")
        self.org_name = Namespace("http://example.org/org_name/")
        self.loc = Namespace("http://example.org/location/")
        self.rel = Namespace("http://example.org/relation/")
        
        # Base classes namespace
        self.base = Namespace("http://example.org/base/")
        
        # Add these relation types explicitly
        self.g = Graph()
        self.g.add((self.rel.hasPosition, RDF.type, RDF.Property))
        self.g.add((self.rel.isEmployedBy, RDF.type, RDF.Property))
        self.g.add((self.rel.hasEmployee, RDF.type, RDF.Property))
        self.g.add((self.rel.hasLocationAt, RDF.type, RDF.Property))
        self.g.add((self.rel.isLocationOf, RDF.type, RDF.Property))
    
    def extract_ontology_classes(self, data: List[Dict]) -> Graph:
        """
        Extract ontology classes from annotations and create TTL file
        """
        positions: Set[str] = set()
        org_roles: Set[str] = set()
        org_sub_roles: Dict[str, str] = {}  # sub_role -> parent_role mapping
        loc_types: Set[str] = set()
        
        # Extract all classes from annotations
        for doc in data:
            for annotation in doc["annotations"]:
                for result in annotation.get('result', []):
                    if 'value' in result:
                        value = result['value']
                        label = value["hypertextlabels"][0]
                        text = value["text"]
                    
                        if label == "Person Position":
                            positions.add(text)
                        elif label == "Organization Role":
                            org_roles.add(text)
                        elif label == "Organization Sub-Role":
                            org_sub_roles[text] = None  # Temporarily store sub-role
                        elif label == "Location Type":
                            loc_types.add(text)
        
        # Create ontology graph
        g = Graph()
        
        # Add namespace prefixes
        g.bind("org_role", self.org_role)
        g.bind("org_sub_role", self.org_sub_role)
        g.bind("person_position", self.person_position)
        g.bind("location_type", self.location_type)
        g.bind("base", self.base)
        
        # Define base classes
        g.add((self.base.Person, RDF.type, RDFS.Class))
        g.add((self.base.Organization, RDF.type, RDFS.Class))
        g.add((self.base.Location, RDF.type, RDFS.Class))
        
        # Add position subclasses
        for position in positions:
            position_uri = self.person_position[self._clean_uri(position)]
            g.add((position_uri, RDF.type, RDFS.Class))
            g.add((position_uri, RDFS.subClassOf, self.base.Person))
        
        # Add organization role subclasses
        for role in org_roles:
            role_uri = self.org_role[self._clean_uri(role)]
            g.add((role_uri, RDF.type, RDFS.Class))
            g.add((role_uri, RDFS.subClassOf, self.base.Organization))
        
        # Add organization sub-role subclasses
        for sub_role, parent_role in org_sub_roles.items():
            sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
            g.add((sub_role_uri, RDF.type, RDFS.Class))
            
            if parent_role:
                parent_role_uri = self.org_role[self._clean_uri(parent_role)]
                g.add((sub_role_uri, RDFS.subClassOf, parent_role_uri))
            else:
                g.add((sub_role_uri, RDFS.subClassOf, self.base.Organization))
        
        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.location_type[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.base.Location))
        
        return g
    
    def create_data_layer(self, doc: Dict, ontology_graph: Graph) -> Graph:
        """
        Create data layer graph for a single document using ontology classes
        """
        g = Graph()

        # Bind namespaces
        g.bind("person_name", self.person_name)
        g.bind("org_name", self.org_name)
        g.bind("loc", self.loc)
        g.bind("base", self.base)
        g.bind("person_position", self.person_position)
        g.bind("org_role", self.org_role)
        g.bind("org_sub_role", self.org_sub_role)
        g.bind("location_type", self.location_type)

        # Define the IsinstanceOf relation
        IsinstanceOf = URIRef(self.base + "IsinstanceOf")
        g.bind("IsinstanceOf", IsinstanceOf)

        # Merge the ontology into the data layer graph
        g += ontology_graph

        entities = {}
        org_roles: Dict[URIRef, Set[URIRef]] = {}  # Store organization roles
        person_positions: Dict[URIRef, URIRef] = {}  # Store person positions
        person_employers: Dict[URIRef, URIRef] = {}  # Store person employers
        location_types: Dict[URIRef, URIRef] = {}  # Store location types

        # First pass: Parse all entities
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "hypertextlabels":
                    label = result["value"].get("hypertextlabels", [])[0]
                    text = result["value"].get("text", "")
                    
                    # Create entity URI
                    uri = None
                    if label == "Person Name":
                        uri = self.person_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Person))  # Link entity to Person class
                        entities[result["id"]] = {"uri": uri, "label": label}
                    
                    elif label == "Organization Name":
                        uri = self.org_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Organization))  # Link entity to Organization class
                        entities[result["id"]] = {"uri": uri, "label": label}
                    
                    elif label == "Location":
                        uri = self.loc[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Location))  # Link entity to Location class
                        entities[result["id"]] = {"uri": uri, "label": label}
                    
                    elif label == "Location Type":
                        uri = self.location_type[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.LocationType))  # Link Location Type to its class
                        entities[result["id"]] = {"uri": uri, "label": label}

                        # Store the location type for later use
                        location_types[result["id"]] = uri
                        
                        # Avoid adding rel.hasLocationType here as per your requirement
                        location_uri = self.loc[self._clean_uri(text)]
                        g.add((location_uri, self.rel.isLocationOf, uri))  # Link Location -> Location Type
                        
                    # Store these for relationship processing
                    elif label in ["Person Position", "Organization Role", "Organization Sub-Role"]:
                        entities[result["id"]] = {
                            "text": text,
                            "label": label
                        }

        # Second pass: Process relationships
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "relation":
                    from_id = result.get("from_id")
                    to_id = result.get("to_id")
                    
                    from_entity = entities.get(from_id)
                    to_entity = entities.get(to_id)
                    
                    # Ensure that the entity has a 'uri' key
                    if not from_entity or not to_entity:
                        continue

                    from_uri = from_entity.get("uri")
                    to_uri = to_entity.get("uri")

                    # If either entity doesn't have a URI, skip processing
                    if not from_uri or not to_uri:
                        continue
                    
                    from_label = from_entity.get("label")
                    to_label = to_entity.get("label")

                    # Define relationships for Person - Position
                    if (from_label == "Person Name" and to_label == "Person Position") or \
                    (from_label == "Person Position" and to_label == "Person Name"):
                        person_uri = from_uri if from_label == "Person Name" else to_uri
                        position_text = to_entity["text"] if from_label == "Person Name" else from_entity["text"]
                        position_uri = self.person_position[self._clean_uri(position_text)]
                        person_positions[person_uri] = position_uri
                        g.add((person_uri, self.rel.hasPosition, position_uri))

                    # Define Organization - Location
                    elif from_label == "Organization Name" and to_label == "Location":
                        org_uri = from_uri
                        loc_uri = to_uri
                        g.add((org_uri, self.rel.hasLocationAt, loc_uri))
                        g.add((loc_uri, self.rel.isLocationOf, org_uri))
                        
                    # Add Position -> Organization relationship (Is Employed By)
                    if (from_label == "Person Position" and to_label == "Organization Name") or \
                    (from_label == "Organization Name" and to_label == "Person Position"):
                        position_uri = from_uri if from_label == "Person Position" else to_uri
                        org_uri = to_uri if from_label == "Person Position" else from_uri
                        g.add((position_uri, self.rel.isEmployedBy, org_uri))
                        g.add((org_uri, self.rel.hasEmployee, position_uri))

        return g

    def _clean_uri(self, text: str) -> str:
        """
        Clean and encode the string text for use as part of a URI.
        """
        return urllib.parse.quote(text.strip().replace(" ", "_"))
    
    def save_graph(self, graph: Graph, filepath: str):
        """
        Save graph to TTL file
        """
        graph.serialize(destination=filepath, format="turtle")


def main(json_file_path: str, output_dir: str):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    builder = KnowledgeGraphBuilder()
    
    # Create and save ontology
    ontology_graph = builder.extract_ontology_classes(data)
    builder.save_graph(ontology_graph, f"{output_dir}/ontology.ttl")
    
    # Create and save data layers
    for doc in data:
        data_graph = builder.create_data_layer(doc, ontology_graph)
        # Use the document ID for naming
        doc_id = doc.get("id", "unknown_id")  # Default to "unknown_id" if not found
        builder.save_graph(data_graph, f"{output_dir}/{doc_id}.ttl")


if __name__ == "__main__":
    # Replace with your actual JSON file path and desired output directory
    main("/Users/vidhyakshayakannan/Documents/vidhyakshaya_corrected_modified.json", "./extracted_content")  
