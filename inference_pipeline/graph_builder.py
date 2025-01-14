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
        g = Graph()
        g.add((self.rel.hasPosition, RDF.type, RDF.Property))
        g.add((self.rel.isEmployedBy, RDF.type, RDF.Property))
        g.add((self.rel.hasEmployee, RDF.type, RDF.Property))
        
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
                            # Store sub-role temporarily
                            org_sub_roles[text] = None
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
        
        # Process relations to find sub-role relationships
        for item in data:
            for result in item.get('annotations', [])[0].get('result', []):
                if result['type'] == 'relation':
                    from_id = result['from_id']
                    to_id = result['to_id']
                    
                    # Get the corresponding annotations
                    from_entity = next((r for r in item['annotations'][0]['result'] 
                                    if r['id'] == from_id and r['type'] == 'hypertextlabels'), None)
                    to_entity = next((r for r in item['annotations'][0]['result'] 
                                    if r['id'] == to_id and r['type'] == 'hypertextlabels'), None)
                    
                    if from_entity and to_entity:
                        from_label = from_entity['value']['hypertextlabels'][0]
                        to_label = to_entity['value']['hypertextlabels'][0]
                        
                        if from_label == "Organization Role" and to_label == "Organization Sub-Role":
                            parent_role = from_entity['value']['text']
                            sub_role = to_entity['value']['text']
                            org_sub_roles[sub_role] = parent_role
        
        # Add organization sub-role subclasses with proper parent relationships
        for sub_role, parent_role in org_sub_roles.items():
            sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
            g.add((sub_role_uri, RDF.type, RDFS.Class))
            
            if parent_role:
                parent_role_uri = self.org_role[self._clean_uri(parent_role)]
                g.add((sub_role_uri, RDFS.subClassOf, parent_role_uri))
            else:
                # If no parent role is found, make it a subclass of Organization
                g.add((sub_role_uri, RDFS.subClassOf, self.base.Organization))
        
        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.location_type[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.base.Location))
        
        return g
     

    def create_data_layer(self, doc: Dict) -> Graph:
        """
        Create data layer graph for a single document using ontology classes
        """
        g = Graph()
        
        # Bind namespaces
        g.bind("person_name", self.person_name)
        g.bind("org_name", self.org_name)
        g.bind("loc", self.loc)
        g.bind("rel", self.rel)
        g.bind("base", self.base)
        g.bind("person_position", self.person_position)
        g.bind("org_role", self.org_role)
        g.bind("org_sub_role", self.org_sub_role)
        g.bind("location_type", self.location_type)
        
        entities = {}
        org_roles: Dict[URIRef, Set[URIRef]] = {}  # Store organization roles
        person_positions: Dict[URIRef, URIRef] = {}  # Store person positions
        position_orgs: Dict[URIRef, URIRef] = {}    # New: Store position -> organization mapping
        person_employers: Dict[URIRef, URIRef] = {}  # Store person employers
        location_types: Dict[URIRef, URIRef] = {}  # Store location types
        
        # First pass: Parse all entities
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "hypertextlabels":
                    label = result["value"].get("hypertextlabels", [])[0]
                    text = result["value"].get("text", "")
                    
                    if label == "Person Name":
                        uri = self.person_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Person))
                        entities[result["id"]] = {"uri": uri, "label": label}
                        person_positions[uri] = None  # Initialize position
                        person_employers[uri] = None  # Initialize employer
                    
                    elif label == "Organization Name":
                        uri = self.org_name[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Organization))
                        org_roles[uri] = set()  # Initialize empty set for roles
                        entities[result["id"]] = {"uri": uri, "label": label}
                    
                    elif label == "Location":
                        uri = self.loc[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.base.Location))
                        entities[result["id"]] = {"uri": uri, "label": label}
                    
                    # Store these for relationship processing
                    elif label in ["Person Position", "Organization Role", "Organization Sub-Role", "Location Type"]:
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
                    
                    if not from_entity or not to_entity:
                        continue
                    
                    from_label = from_entity.get("label")
                    to_label = to_entity.get("label")
                    
                    # Person - Position relationship
                    if (from_label == "Person Name" and to_label == "Person Position") or \
                       (from_label == "Person Position" and to_label == "Person Name"):
                        person_uri = from_entity["uri"] if from_label == "Person Name" else to_entity["uri"]
                        position_text = to_entity["text"] if from_label == "Person Name" else from_entity["text"]
                        position_uri = self.person_position[self._clean_uri(position_text)]
                        person_positions[person_uri] = position_uri
                        # Add position relationship without duplicate type
                        g.add((person_uri, RDF.type, self.base.Person))
                        # g.add((person_uri, self.rel.hasPosition, position_uri))
                    
                    # Organization - Person relationship
                    elif (from_label == "Organization Name" and to_label == "Person Name") or \
                         (from_label == "Person Name" and to_label == "Organization Name"):
                        org_uri = from_entity["uri"] if from_label == "Organization Name" else to_entity["uri"]
                        person_uri = to_entity["uri"] if from_label == "Organization Name" else from_entity["uri"]
                        
                        # Add employment relationships
                        g.add((person_uri, RDF.type, self.base.Person))
                        g.add((org_uri, RDF.type, self.base.Organization))
                        g.add((person_uri, self.rel.isEmployedBy, org_uri))
                        g.add((org_uri, self.rel.hasEmployee, person_uri))
                        person_employers[person_uri] = org_uri
                    
                    # Organization - Role relationship
                    elif (from_label == "Organization Name" and to_label in ["Organization Role", "Organization Sub-Role"]):
                        org_uri = from_entity["uri"]
                        role_ns = self.org_role if to_label == "Organization Role" else self.org_sub_role
                        role_uri = role_ns[self._clean_uri(to_entity["text"])]
                        g.add((org_uri, RDF.type, self.base.Organization))
                        # g.add((org_uri, RDF.type, role_uri))
                        # org_roles[org_uri].add(role_uri)
                    
                    # Organization - Sub-Role relationship
                    elif from_label == "Organization Name" and to_label == "Organization Sub-Role":
                        org_uri = from_entity["uri"]
                        sub_role_uri = self.org_sub_role[self._clean_uri(to_entity["text"])]
                        org_roles[org_uri].add(sub_role_uri)
                    
                    # Location - Type relationship
                    elif from_label == "Location" and to_label == "Location Type":
                        loc_uri = from_entity["uri"]
                        type_uri = self.location_type[self._clean_uri(to_entity["text"])]
                        # g.add((loc_uri, RDF.type, type_uri))
                    
                    # Organization - Location relationship
                    elif from_label == "Organization Name" and to_label == "Location":
                        org_uri = from_entity["uri"]
                        loc_uri = to_entity["uri"]
                        g.add((org_uri, self.rel.hasLocationAt, loc_uri))
                        g.add((loc_uri, self.rel.isLocationOf, org_uri))
                    
                    # Add Position -> Organization relationship handling
                    if (from_label == "Person Position" and to_label == "Organization Name") or \
                       (from_label == "Organization Name" and to_label == "Person Position"):
                        org_uri = from_entity["uri"] if from_label == "Organization Name" else to_entity["uri"]
                        position_text = to_entity["text"] if from_label == "Organization Name" else from_entity["text"]
                        position_uri = self.person_position[self._clean_uri(position_text)]
                        position_orgs[position_uri] = org_uri
        
        # After processing all relationships, connect Person to Organization through Position
        for person_uri, position_uri in person_positions.items():
            if position_uri in position_orgs:
                org_uri = position_orgs[position_uri]
                g.add((person_uri, self.rel.isEmployedBy, org_uri))
                g.add((org_uri, self.rel.hasEmployee, person_uri))
        
        # Add any missing relationships
        for person_uri, position_uri in person_positions.items():
            if position_uri and person_uri not in [s for s,p,o in g.triples((None, self.rel.hasPosition, position_uri))]:
                g.add((person_uri, RDF.type, self.base.Person))
                # g.add((person_uri, RDF.type, position_uri))
                # g.add((person_uri, self.rel.hasPosition, position_uri))

        for person_uri, employer_uri in person_employers.items():
            if employer_uri and person_uri not in [s for s,p,o in g.triples((None, self.rel.isEmployedBy, employer_uri))]:
                g.add((person_uri, self.rel.isEmployedBy, employer_uri))
                g.add((employer_uri, self.rel.hasEmployee, person_uri))
        
        # Add all organization roles
        for org_uri, roles in org_roles.items():
            for role_uri in roles:
                g.add((org_uri, RDF.type, role_uri))
        
        return g
    
   
    def _clean_uri(self, text: str) -> str:
        """
        Clean text for use in URIs
        """
        return urllib.parse.quote(text.strip().replace("\n", "").replace(" ", "_"), safe="_")
    
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
        data_graph = builder.create_data_layer(doc)
        # Use the document ID for naming
        doc_id = doc.get("id", "unknown_id")  # Default to "unknown_id" if not found
        builder.save_graph(data_graph, f"{output_dir}/{doc_id}.ttl")


if __name__ == "__main__":
    # Replace with your actual JSON file path and desired output directory
    main("48 to 53_hari.json", "./extracted_content")    