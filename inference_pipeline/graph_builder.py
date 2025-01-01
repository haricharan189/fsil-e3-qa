"""
graph_builder.py

Changes:
 - Reads config for JSON_FILE_PATH and EXTRACTED_CONTENT_DIR
 - Writes TTL files to extracted_content
 - Comments for clarity
"""

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import urllib.parse
import os

import config

class KnowledgeGraphBuilder:
    def __init__(self):
        self.org    = Namespace("http://example.org/organization/")
        self.person = Namespace("http://example.org/person/")
        self.loc    = Namespace("http://example.org/location/")
        self.rel    = Namespace("http://example.org/relation/")
        
    def extract_ontology_classes(self, data: List[Dict]) -> Graph:
        positions: Set[str]     = set()
        org_roles: Set[str]     = set()
        org_sub_roles: Set[str] = set()
        loc_types: Set[str]     = set()
        
        # Gather classes from JSON
        for doc in data:
            for annotation in doc["annotations"]:
                for result in annotation.get('result', []):
                    if 'value' in result:
                        value = result['value']
                        label = value["hypertextlabels"][0]
                        text  = value["text"]
                    
                        if label == "Person Position":
                            positions.add(text)
                        elif label == "Organization Role":
                            org_roles.add(text)
                        elif label == "Organization Sub-Role":
                            org_sub_roles.add(text)
                        elif label == "Location Type":
                            loc_types.add(text)
        
        g = Graph()
        g.bind("org", self.org)
        g.bind("person", self.person)
        g.bind("loc", self.loc)
        g.bind("rel", self.rel)
        
        # Define top-level classes
        g.add((self.person.Person, RDF.type, RDFS.Class))
        g.add((self.org.Organization, RDF.type, RDFS.Class))
        g.add((self.loc.Location, RDF.type, RDFS.Class))
        
        # Subclasses
        for position in positions:
            position_uri = self.person[self._clean_uri(position)]
            g.add((position_uri, RDF.type, RDFS.Class))
            g.add((position_uri, RDFS.subClassOf, self.person.Person))
            
        for role in org_roles:
            role_uri = self.org[self._clean_uri(role)]
            g.add((role_uri, RDF.type, RDFS.Class))
            g.add((role_uri, RDFS.subClassOf, self.org.Organization))
            
        for sub_role in org_sub_roles:
            sub_role_uri = self.org[self._clean_uri(sub_role)]
            g.add((sub_role_uri, RDF.type, RDFS.Class))
            g.add((sub_role_uri, RDFS.subClassOf, self.org.Organization))

        # Attempt to find relationships for sub-roles
        entities = {}
        relations_with_labels = []

        for item in data:
            # If no annotations in an item, skip safely
            if "annotations" not in item or not item["annotations"]:
                continue
            for result in item["annotations"][0].get('result', []):
                if result['type'] == 'hypertextlabels':
                    entity_id = result['id']
                    label     = result['value']['hypertextlabels'][0]
                    text      = result['value']['text']
                    # For simplicity, if label starts with "Organization", treat it as org
                    if label.startswith("Organization"):
                        uri = self.org[self._clean_uri(text)]
                    else:
                        uri = self.loc[self._clean_uri(text)]
                    entities[entity_id] = {'label': label, 'text': text, 'uri': uri}

        for item in data:
            if "annotations" not in item or not item["annotations"]:
                continue
            for result in item["annotations"][0].get('result', []):
                if result['type'] == 'relation':
                    from_id = result['from_id']
                    to_id   = result['to_id']
                    from_entity = entities.get(from_id)
                    to_entity   = entities.get(to_id)
                    if from_entity and to_entity:
                        relations_with_labels.append({
                            'from': from_entity,
                            'to':   to_entity
                        })

        # Update sub-role relationships
        for relation in relations_with_labels:
            if (relation['from']['label'] == "Organization Role" 
                and relation['to']['label'] == "Organization Sub-Role"):
                sub_role_uri = relation['to']['uri']
                role_uri     = relation['from']['uri']
                # Remove default subClassOf org:Organization, replace with the new role
                g.remove((sub_role_uri, RDFS.subClassOf, self.org.Organization))
                g.add((sub_role_uri, RDFS.subClassOf, role_uri))
            
        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.loc[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.loc.Location))
            
        return g
     
    def create_data_layer(self, doc: Dict) -> Graph:
        """
        For each doc in the JSON, create a data-layer graph.
        """
        g = Graph()
        g.bind("org", self.org)
        g.bind("person", self.person)
        g.bind("loc", self.loc)
        g.bind("rel", self.rel)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        
        entities = {}
        position_to_org: Dict[str, URIRef]    = {}
        position_to_person: Dict[str, URIRef] = {}
        org_roles: Dict[URIRef, List[URIRef]] = {}

        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "hypertextlabels":
                    label = result["value"].get("hypertextlabels", [])[0]
                    text  = result["value"].get("text", "")
                    uri   = None
                    
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
                        uri = self.org[self._clean_uri(text)]
                    
                    if uri:
                        entities[result["id"]] = {"uri": uri, "label": label, "text": text}
        
        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "relation":
                    from_id = result.get("from_id")
                    to_id   = result.get("to_id")
                    from_entity = entities.get(from_id)
                    to_entity   = entities.get(to_id)
                    if not from_entity or not to_entity:
                        continue
                    
                    from_label = from_entity["label"]
                    to_label   = to_entity["label"]
                    from_uri   = from_entity["uri"]
                    to_uri     = to_entity["uri"]
                    
                    # Organization -> Role
                    if from_label == "Organization Name" and to_label == "Organization Role":
                        if from_uri not in org_roles:
                            org_roles[from_uri] = []
                        org_roles[from_uri].append(to_uri)
                    # Organization -> Location
                    elif from_label == "Organization Name" and to_label == "Location":
                        g.add((from_uri, self.rel.hasLocationAt, to_uri))
                        g.add((to_uri, self.rel.isLocationOf, from_uri))
                    # Organization -> Person Position
                    if from_label == "Organization Name" and to_label == "Person Position":
                        position_text = to_entity["text"]
                        position_to_org[position_text] = from_uri
                    # Person Position -> Person Name
                    elif from_label == "Person Position" and to_label == "Person Name":
                        position_text = from_entity["text"]
                        position_to_person[position_text] = to_uri
    
        # Link Persons to Orgs
        for position_text, org_uri in position_to_org.items():
            person_uri = position_to_person.get(position_text)
            if person_uri:
                position_class_uri = self.person[self._clean_uri(position_text)]
                g.add((person_uri, RDF.type, position_class_uri))
                
                g.add((person_uri, self.rel.isEmployedBy, org_uri))
                g.add((org_uri, self.rel.hasEmployee, person_uri))
        
        # Add roles to org
        for org_uri, roles in org_roles.items():
            for role_uri in roles:
                g.add((org_uri, RDF.type, role_uri))
        
        return g
    
    def _clean_uri(self, text: str) -> str:
        return urllib.parse.quote(text.strip().replace("\n", "").replace(" ", "_"), safe="_")
    
    def save_graph(self, graph: Graph, filepath: str):
        graph.serialize(destination=filepath, format="turtle")

def main():
    Path(config.EXTRACTED_CONTENT_DIR).mkdir(parents=True, exist_ok=True)
    with open(config.JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    builder = KnowledgeGraphBuilder()
    
    # 1) Ontology
    ontology_graph = builder.extract_ontology_classes(data)
    ontology_ttl   = os.path.join(config.EXTRACTED_CONTENT_DIR, "ontology.ttl")
    builder.save_graph(ontology_graph, ontology_ttl)
    
    # 2) Data layer
    for doc in data:
        data_graph = builder.create_data_layer(doc)
        doc_id     = doc.get("id", "unknown_id")
        output_ttl = os.path.join(config.EXTRACTED_CONTENT_DIR, f"{doc_id}.ttl")
        builder.save_graph(data_graph, output_ttl)

if __name__ == "__main__":
    main()
