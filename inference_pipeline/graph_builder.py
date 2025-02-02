from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import urllib.parse
import os
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure necessary NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

class KnowledgeGraphBuilder:
    def __init__(self):
        """Initialize the knowledge graph with namespaces and properties."""
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
        
        # Define isInstanceOf as its own predicate
        self.isInstanceOf = URIRef("http://example.org/isInstanceOf/")
        
        # Initialize RDF graph
        self.g = Graph()
        
        # Add predefined relation types to the graph
        self.g.add((self.rel.hasPosition, RDF.type, RDF.Property))
        self.g.add((self.rel.isEmployedBy, RDF.type, RDF.Property))
        self.g.add((self.rel.hasEmployee, RDF.type, RDF.Property))
        self.g.add((self.isInstanceOf, RDF.type, RDF.Property))
        
        # Initialize lemmatizer
        self._lemmatizer = WordNetLemmatizer()
    

    def lemmatize_text(self, text: str) -> str:
        """Remove periods, normalize spaces, and lemmatize a given text (for nouns)."""
        # Remove periods
        text = re.sub(r'\.', '', text)  # Remove periods
        
        # Normalize spaces: replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with one, and strip leading/trailing spaces
        
        # Convert to lowercase and split into words
        words = text.lower().split()  # Convert to lowercase and split into words
        
        # Lemmatize words as nouns
        lemmatized_words = [self._lemmatizer.lemmatize(word, wordnet.NOUN) for word in words]
        
        # Join back the lemmatized words
        return " ".join(lemmatized_words)


    def generate_role_subrole_map(self, data: List[Dict]) -> List[tuple[str, str]]:
        """Extracts role-subrole relationships and returns them as tuples."""
        role_subrole_pairs = []  # List to store role-subrole pairs

        # First pass: Extract only the role-subrole relationships
        for doc in data:
            entities = {}  # Temporary store for entities in the current document
            
            # Parse all entities (roles and sub-roles)
            for annotation in doc.get("annotations", []):
                for result in annotation.get("result", []):
                    if "value" in result:
                        value = result["value"]
                        label = value.get("hypertextlabels", [])[0]
                        text = value.get("text", "")

                        # Lemmatize text before storing it
                        lemmatized_text = self.lemmatize_text(text)

                        # Store Organization Role and Sub-Role entities
                        if label in ["Organization Role", "Organization Sub-Role"]:
                            entities[result["id"]] = {
                                "text": lemmatized_text,  # Store lemmatized text
                                "label": label
                            }
                            
            # Second pass: role-subrole relationships
            for annotation in doc.get("annotations", []):
                for result in annotation.get("result", []):
                    if result["type"] == "relation":
                        from_id = result.get("from_id")
                        to_id = result.get("to_id")

                        # Retrieve entities based on IDs
                        from_entity = entities.get(from_id)
                        to_entity = entities.get(to_id)

                        if not from_entity or not to_entity:
                            continue
                        
                        from_label = from_entity.get("label")
                        to_label = to_entity.get("label")

                        # If we have an Organization Role - Organization Sub-Role relationship
                        if from_label == "Organization Role" and to_label == "Organization Sub-Role":
                            role_uri = from_entity["text"]
                            sub_role_uri = to_entity["text"]

                            role_subrole_pairs.append((role_uri, sub_role_uri))  # Add to the list

        return role_subrole_pairs
        
    def extract_ontology_classes(self, data: List[Dict], role_subrole_pairs: List[tuple[str, str]]) -> Graph:
        """
        Extract ontology classes from annotations and create TTL file.
        """
        positions: Set[str] = set()
        org_roles: Set[str] = set()
        org_sub_roles: Set[str] = set()
        loc_types: Set[str] = set()

        # Extract all classes from annotations
        for doc in data:
            for annotation in doc.get("annotations", []):
                for result in annotation.get("result", []):
                    if "value" in result:
                        value = result["value"]
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
        
        for role in org_roles:
                role_uri = self.org_role[self._clean_uri(role)]
                g.add((role_uri, RDF.type, RDFS.Class))
                g.add((role_uri, RDFS.subClassOf, self.base.Organization))

                # Add corresponding sub-roles, only if they exist in the map
                if role in role_subrole_pairs:
                    for sub_role in role_subrole_pairs[role]:
                        sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
                        g.add((sub_role_uri, RDF.type, RDFS.Class))
                        g.add((sub_role_uri, RDFS.subClassOf, role_uri))

        # Add location type subclasses
        for loc_type in loc_types:
            loc_type_uri = self.location_type[self._clean_uri(loc_type)]
            g.add((loc_type_uri, RDF.type, RDFS.Class))
            g.add((loc_type_uri, RDFS.subClassOf, self.base.Location))

        # Add organization sub-role subclasses based on role-subrole pairs
        for role, sub_role in role_subrole_pairs:
            sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
            role_uri = self.org_role[self._clean_uri(role)]
            g.add((sub_role_uri, RDFS.subClassOf, role_uri))

        return g
     

    def create_data_layer(self, doc: Dict, ontology_graph) -> Graph:
        
        """
        Create data layer graph for a single document using ontology classes
        """
        g = Graph()
        
        g = g + ontology_graph

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
        position_orgs: Dict[URIRef, URIRef] = {}    # Store position -> organization mapping
        person_employers: Dict[URIRef, URIRef] = {}  # Store person employers
        org_names:Dict[URIRef, URIRef] = {}
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
                        g.add((person_uri, self.isInstanceOf, position_uri))

                    # Organization - Role relationship
                    elif (from_label == "Organization Name" and to_label in ["Organization Role", "Organization Sub-Role"]):
                        org_uri = from_entity["uri"]
                        if(to_label == "Organization Role"):
                            role_ns = self.org_role
                            role_uri = role_ns[self._clean_uri(to_entity["text"])]
                            g.add((org_uri, self.isInstanceOf, role_uri))
                        else:
                            role_ns = self.org_sub_role
                            sub_role_uri = role_ns[self._clean_uri(to_entity["text"])]
                            g.add((org_uri, self.isInstanceOf, sub_role_uri))

                        g.add((org_uri, RDF.type, self.base.Organization))
                        # g.add((org_uri, RDF.type, role_uri))
                        # org_roles[org_uri].add(role_uri)
                    
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
                        g.add((person_uri, self.isInstanceOf, org_uri))
                        person_employers[person_uri] = org_uri
                   
                    # Location - Type relationship
                    elif from_label == "Location" and to_label == "Location Type":
                        loc_uri = from_entity["uri"]
                        type_uri = self.location_type[self._clean_uri(to_entity["text"])]
                        g.add((loc_uri, self.isInstanceOf, type_uri))
                    
                    # Organization - Location relationship
                    elif from_label == "Organization Name" and to_label == "Location":
                        org_uri = from_entity["uri"]
                        loc_uri = to_entity["uri"]
                        g.add((org_uri, self.rel.hasLocationAt, loc_uri))
                        g.add((loc_uri, self.rel.isLocationOf, org_uri))
                        g.add((loc_uri, self.isInstanceOf, self.base.Location))
                    
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
            if position_uri and person_uri not in [s for s, p, o in g.triples((None, self.rel.hasPosition, position_uri))]:
                g.add((person_uri, RDF.type, self.base.Person))
                g.add((person_uri, self.isInstanceOf, position_uri))

        for person_uri, employer_uri in person_employers.items():
            if employer_uri and person_uri not in [s for s, p, o in g.triples((None, self.rel.isEmployedBy, employer_uri))]:
                g.add((person_uri, self.rel.isEmployedBy, employer_uri))
                g.add((employer_uri, self.rel.hasEmployee, person_uri))
                
   
        entities = {}  # Temporary store for entities in the current document
        # First pass: Parse all entities (roles, sub-roles, and organization names)
        for annotation in doc.get("annotations", []):
            for result in annotation.get("result", []):
                if "value" in result:
                    value = result["value"]
                    label = value.get("hypertextlabels", [])[0]
                    text = value.get("text", "")

                    # Store Organization Role, Sub-Role, and Organization Name entities
                    if label in ["Organization Role", "Organization Sub-Role", "Organization Name"]:
                        entities[result["id"]] = {
                            "text": text,
                            "label": label
                        }

        # Second pass: Process relationships
        print("Processing relationships...")
        role_to_subrole = {}  # Map roles to their sub-roles
        org_to_role = {}  # Map organizations to their roles

        for annotation in doc.get("annotations", []):
            for result in annotation.get('result', []):
                if result["type"] == "relation":
                    from_id = result.get("from_id")
                    to_id = result.get("to_id")
                    from_entity = entities.get(from_id)
                    to_entity = entities.get(to_id)

                    if not from_entity or not to_entity:
                        print(f"Skipping relationship: from_id={from_id}, to_id={to_id} (one or both not found)")
                        continue

                    from_label = from_entity.get("label")
                    to_label = to_entity.get("label")

                    print(f"Processing relationship: from_label={from_label}, to_label={to_label}")

                    # Track role-to-subrole mapping
                    if from_label == "Organization Role" and to_label == "Organization Sub-Role":
                        role = from_entity["text"]
                        sub_role = to_entity["text"]
                        role_to_subrole.setdefault(role, set()).add(sub_role)
                        print(f"Mapped role to sub-role: role={role}, sub_role={sub_role}")

                    # Track organization-to-role mapping
                    elif (from_label == "Organization Name" and to_label == "Organization Role") or \
                        (from_label == "Organization Role" and to_label == "Organization Name"):
                        org = from_entity["text"] if from_label == "Organization Name" else to_entity["text"]
                        role = to_entity["text"] if from_label == "Organization Name" else from_entity["text"]
                        org_to_role.setdefault(org, set()).add(role)
                        print(f"Mapped organization to role: org={org}, role={role}")

        # Final pass: Determine isInstanceOf relationships
        print("Assigning isInstanceOf relationships...")
        for org, roles in org_to_role.items():
            for role in roles:
                sub_roles = role_to_subrole.get(role, None)  # Get sub-roles for this role
                if sub_roles:  # Role has sub-roles
                    for sub_role in sub_roles:
                        org_uri = self.org_name[self._clean_uri(org)]
                        sub_role_uri = self.org_sub_role[self._clean_uri(sub_role)]
                        g.add((org_uri, self.isInstanceOf, sub_role_uri))
                        print(f"{org} isInstanceOf {sub_role}")
                else:  # Role does not have sub-roles
                    org_uri = self.org_name[self._clean_uri(org)]
                    role_uri = self.org_role[self._clean_uri(role)]
                    g.add((org_uri, self.isInstanceOf, role_uri))
                    print(f"{org} isInstanceOf {role}")

        return g
   
    def _clean_uri(self, text: str, is_role_subrole: bool = False) -> str:
        """
        Clean text for use in URIs
        """
        cleaned = urllib.parse.quote(text.strip().replace("\n", "").replace(" ", "_"), safe="_")
        if is_role_subrole:
            cleaned = " ".join(self._lemmatizer.lemmatize(word.lower()) for word in cleaned.split())
        return cleaned
    
    def save_graph(self, graph: Graph, filepath: str):
        """
        Save graph to TTL file
        """
        graph.serialize(destination=filepath, format="turtle")


def main(json_file_path: str, output_dir: str):
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        builder = KnowledgeGraphBuilder()

        # Load JSON data
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        role_subrole_map = builder.generate_role_subrole_map(data)
        # Create and save ontology
        ontology_graph = builder.extract_ontology_classes(data, role_subrole_map)
        builder.save_graph(ontology_graph, f"{output_dir}/ontology.ttl")

        # Create and save data layers
        for doc in data:
            data_graph = builder.create_data_layer(doc, ontology_graph)
            # Use the document ID for naming
            doc_id = doc.get("id", "unknown_id")
            builder.save_graph(data_graph, f"{output_dir}/{doc_id}.ttl")



if __name__ == "__main__":
    # Replace with your actual JSON file path and desired output directory
    main("/Users/vidhyakshayakannan/Downloads/semi_cleaned_docs.json", "./extracted_content")    
