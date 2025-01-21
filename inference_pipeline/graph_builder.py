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
        self.is_instance_of = URIRef("http://example.org/is_instance_of/")
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
        g.bind("is_instance_of", self.is_instance_of)

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
        Create data layer graph for a single document using ontology classes.
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
        g.bind("rel", self.rel)

        # Merge the ontology into the data layer graph
        g += ontology_graph

        entities = {}

        # Parse all entities
        for annotation in doc.get("annotations", []):
            for result in annotation.get("result", []):
                if result["type"] == "hypertextlabels":
                    label = result["value"].get("hypertextlabels", [])[0]
                    text = result["value"].get("text", "")

                    # Handle Person Name
                    if label == "Person Name":
                        uri = self.person_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Person))  # Declare as instance of base:Person
                        g.add((uri, self.is_instance_of, self.person_position["VP"]))  # IsInstanceOf VP
                        g.add((uri, self.rel.IsEmployeeOf, self.org_name["Wells_Fargo"]))  # IsEmployeeOf Wells Fargo
                        g.add((uri, RDFS.label, Literal(text)))  # Add label as literal
                        entities[result["id"]] = {"uri": uri, "label": label}

                    # Handle Organization Name
                    elif label == "Organization Name":
                        uri = self.org_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Organization))  # Declare as instance of base:Organization
                        g.add((uri, self.is_instance_of, self.base.Organization))  # IsinstanceOf Organization
                        role_uri = self.org_role[self._clean_uri("Finance_Department")]
                        g.add((uri, self.rel.hasOrgRole, role_uri))  # Link to role
                        entities[result["id"]] = {"uri": uri, "label": label}

                    # Handle Location
                    elif label == "Location":
                        uri = self.loc[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Location))  # Declare as instance of base:Location
                        location_type_uri = self.location_type[self._clean_uri("Loan_and_Agency_Services_Group")]
                        g.add((uri, self.is_instance_of, location_type_uri))  # IsInstanceOf location type
                        g.add(
                            (uri, self.rel.isLocationOf, self.org_name[self._clean_uri("JPMORGAN_CHASE_BANK_N_A")])
                        )  # Link to organization
                        g.add((uri, RDFS.label, Literal(text)))  # Add label as literal
                        entities[result["id"]] = {"uri": uri, "label": label}

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
