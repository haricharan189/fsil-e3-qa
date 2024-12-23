from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import urllib.parse
import os
class KnowledgeGraphBuilder:
    def __init__(self):
        # Define namespaces
        self.org = Namespace("http://example.org/organization/")
        self.person = Namespace("http://example.org/person/")
        self.loc = Namespace("http://example.org/location/")
        self.rel = Namespace("http://example.org/relation/")
        
    def extract_ontology_classes(self, data: List[Dict]) -> Graph:
        """
        Extract ontology classes from annotations and create TTL file
        """
        positions: Set[str] = set()
        org_roles: Set[str] = set()
        org_sub_roles: Set[str] = set()
        loc_types: Set[str] = set()
        
        # Extract all classes from annotations
        for doc in data:
            for annotation in doc["annotations"]:
                for result in annotation.get('result', []):
                    if 'value' in result:
                        value=result['value']
                        label = value["hypertextlabels"][0]
                        text = value["text"]
                    
                        if label == "Person Position":
                           positions.add(text)
                        elif label == "Organization Role":
                             org_roles.add(text)
                        elif label == "Organization Sub-Role":
                             org_sub_roles.add(text)
                        elif label == "Location Type":
                            loc_types.add(text)
        
        # Create ontology graph
        g = Graph()
        
        # Add namespace prefixes
        g.bind("org", self.org)
        g.bind("person", self.person)
        g.bind("loc", self.loc)
        g.bind("rel", self.rel)
        entities ={}
        relations_with_labels = []
        
        # Define main classes
        g.add((self.person.Person, RDF.type, RDFS.Class))
        g.add((self.org.Organization, RDF.type, RDFS.Class))
        g.add((self.loc.Location, RDF.type, RDFS.Class))
        
        # Add position subclasses
        for position in positions:
            position_uri = self.person[self._clean_uri(position)]
            g.add((position_uri, RDF.type, RDFS.Class))
            g.add((position_uri, RDFS.subClassOf, self.person.Person))
            
        # Add organization role subclasses
        for role in org_roles:
            role_uri = self.org[self._clean_uri(role)]
            g.add((role_uri, RDF.type, RDFS.Class))
            g.add((role_uri, RDFS.subClassOf, self.org.Organization))
            
        # Add organization sub-role subclasses
        for sub_role in org_sub_roles:
            sub_role_uri = self.org[self._clean_uri(sub_role)]
            g.add((sub_role_uri, RDF.type, RDFS.Class))
    # Initial link to the Organization class - this will be updated later
            g.add((sub_role_uri, RDFS.subClassOf, self.org.Organization))


        for item in data:
            for result in item.get('annotations', [])[0].get('result', []):
                if result['type'] == 'hypertextlabels':
                    entity_id = result['id']
                    label = result['value']['hypertextlabels'][0]  # e.g., "Organization Name"
                    text = result['value']['text']  # e.g., "RG PARENT LLC"
                    uri = self.org[self._clean_uri(text)] if label.startswith("Organization") else self.loc[self._clean_uri(text)]
                    entities[entity_id] = {'label': label, 'text': text, 'uri': uri}

    # Second pass: Extract relations and match them to the corresponding labels and annotations
        for item in data:
            for result in item.get('annotations', [])[0].get('result', []):
                if result['type'] == 'relation':
                    from_id = result['from_id']
                    to_id = result['to_id']

                # Get corresponding annotations and labels for the 'from' and 'to' entities
                    from_entity = entities.get(from_id, None)
                    to_entity = entities.get(to_id, None)

                    if from_entity and to_entity:
                        relations_with_labels.append({
                            'from': {
                                'id': from_id,
                                'label': from_entity['label'],
                                'text': from_entity['text'],
                                'uri': from_entity['uri']
                            },
                            'to': {
                                'id': to_id,
                                'label': to_entity['label'],
                                'text': to_entity['text'],
                                'uri': to_entity['uri']
                            }
                        })


        for relation in relations_with_labels:
            if relation['from']['label'] == "Organization Role" and relation['to']['label'] == "Organization Sub-Role":
        # Update the subclass relationship
                sub_role_uri = relation['to']['uri']
                role_uri = relation['from']['uri']
        
        # Ensure the sub-role is a subclass of the specific role
                g.remove((sub_role_uri, RDFS.subClassOf, self.org.Organization))  # Remove default subclass relationship
                g.add((sub_role_uri, RDFS.subClassOf, role_uri))  # Add new subclass relationship

            
        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.loc[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.loc.Location))
            
        return g
     

    def create_data_layer(self, doc: Dict) -> Graph:
        """
        Create data layer graph for a single document
        """
        g = Graph()
        g.bind("org", self.org)
        g.bind("person", self.person)
        g.bind("loc", self.loc)
        g.bind("rel", self.rel)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        
        entities = {}
        position_to_org: Dict[str, URIRef] = {}
        position_to_person: Dict[str, URIRef] = {}
        org_roles: Dict[URIRef, List[URIRef]] = {}

        # First pass: Parse all entities
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "hypertextlabels":
                    label = result["value"].get("hypertextlabels", [])[0]
                    text = result["value"].get("text", "")
                    uri = None
                    
                    if label == "Person Name":
                        uri = self.person[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.person.Person))
                    elif label == "Organization Name":
                        uri = self.org[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.org.Organization))
                    elif label == "Location":
                        uri = self.loc[self._clean_uri(text)]
                        g.add((uri, RDF.type, self.loc.Location))
                    elif label == "Person Position":
                        uri = self.person[self._clean_uri(text)]
                        g.add((uri, RDF.type, RDFS.Class)) 
                    elif label == "Organization Role":
                        uri = self.org[self._clean_uri(text)]    # Positions are treated as classes
                    
                    if uri:
                        entities[result["id"]] = {"uri": uri, "label": label, "text": text}
        
        # Second pass: Parse all relations
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "relation":
                    from_id = result.get("from_id")
                    to_id = result.get("to_id")
                    
                    from_entity = entities.get(from_id)
                    to_entity = entities.get(to_id)
                    
                    if not from_entity or not to_entity:
                        continue  # Skip if entities are not found
                    
                    from_label = from_entity["label"]
                    to_label = to_entity["label"]
                    from_uri = from_entity["uri"]
                    to_uri = to_entity["uri"]
                    
                    # Determine the type of relation
                    if from_label == "Organization Name" and to_label == "Organization Role":
                        # Track the roles of the organization
                        if from_uri not in org_roles:
                            org_roles[from_uri] = []
                        org_roles[from_uri].append(to_uri)
                    elif from_label == "Organization Name" and to_label == "Location":
                        # Organization has Location
                        g.add((from_uri, self.rel.hasLocationAt, to_uri))
                        g.add((to_uri, self.rel.isLocationOf, from_uri))
                    if from_label == "Organization Name" and to_label == "Person Position":
                        # Organization has a Position
                        position_text = to_entity["text"]
                        position_to_org[position_text] = from_uri
                    elif from_label == "Person Position" and to_label == "Person Name":
                        # Position is held by Person
                        position_text = from_entity["text"]
                        position_to_person[position_text] = to_uri
    
        # Third pass: Link Persons to Organizations via Positions
        for position_text, org_uri in position_to_org.items():
            person_uri = position_to_person.get(position_text)
            if person_uri:
                # Assign Position as rdf:type to Person
                position_class_uri = self.person[self._clean_uri(position_text)]
                g.add((person_uri, RDF.type, position_class_uri))
                
                # Link Person to Organization
                g.add((person_uri, self.rel.isEmployedBy, org_uri))
                g.add((org_uri,self.rel.hasEmployee, person_uri))
        
        # Fourth pass: Add organization roles
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