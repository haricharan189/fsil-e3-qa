from rdflib import Graph, URIRef, Literal, BNode, RDF, Namespace
from rdflib.namespace import RDF, RDFS
import json
from typing import Set, Dict, List
from pathlib import Path
import re
from collections import defaultdict
from urllib.parse import urlparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

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
        """Remove periods, normalize spaces, and lemmatize nouns in the text."""
        # Remove periods
        text = re.sub(r'\.', '', text)  # Remove periods
        text = re.sub(r'\,', '', text)  
        
        # Normalize spaces: replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()  
        words = nltk.word_tokenize(text.lower())  
        # POS tagging for each word
        pos_tags = nltk.pos_tag(words)
        
        # Lemmatize words based on their part of speech
        lemmatized_words = [
            self._lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(tag)) if self.get_wordnet_pos(tag) else word
            for word, tag in pos_tags
        ]
        
        # Join back the lemmatized words into a single string
        return " ".join(lemmatized_words)
    
    def get_wordnet_pos(self, treebank_tag: str) -> str:
        """Map POS tag to WordNet POS format."""
        if treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('R'):
            return wn.ADV
        elif treebank_tag.startswith('J'):
            return wn.ADJ
        else:
            return None


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

        # 1) Collect all class‐labels from the annotations
        for doc in data:
            for annotation in doc.get("annotations", []):
                for result in annotation.get("result", []):
                    if "value" in result:
                        label = result["value"]["hypertextlabels"][0]
                        text  = result["value"]["text"].strip()

                        if label == "Person Position":
                            positions.add(text)
                        elif label == "Organization Role":
                            org_roles.add(text)
                        elif label == "Organization Sub-Role":
                            org_sub_roles.add(text)
                        elif label == "Location Type":
                            loc_types.add(text)

        # 2) Build the ontology graph
        g = Graph()

        # Bind prefixes
        g.bind("org_role",      self.org_role)
        g.bind("org_sub_role",  self.org_sub_role)
        g.bind("person_position", self.person_position)
        g.bind("location_type", self.location_type)
        g.bind("base",          self.base)

        # Declare the base classes
        g.add((self.base.Person,       RDF.type, RDFS.Class))
        g.add((self.base.Organization, RDF.type, RDFS.Class))
        g.add((self.base.Location,     RDF.type, RDFS.Class))

        # 3) Add each Location Type as a subclass of base:Location
        for text in loc_types:
            cleaned = self._clean_uri(text)
            uri = self.location_type[cleaned]
            g.add((uri, RDF.type,       RDFS.Class))
            g.add((uri, RDFS.subClassOf, self.base.Location))

        return g

    
    def create_data_layer(self, doc: Dict, ontology_graph: Graph) -> Graph:
        log_lines = []
        def log(msg: str):
            print(msg)
            log_lines.append(msg)

        g = Graph()
        g += ontology_graph

        # Bind prefixes
        prefixes = [
            ('base', self.base),
            ('person', self.person_name),
            ('org', self.org_name),
            ('org_role', self.org_role),
            ('org_sub_role', self.org_sub_role),
            ('position', self.person_position),
            ('loc', self.loc),
            ('location_type', self.location_type),
            ('rel', self.rel),
        ]

        def bind_prefixes():
            for pfx, ns in prefixes:
                g.bind(pfx, ns)
        bind_prefixes()

        registry = {}
        deferred_positions = []
        org_roles = defaultdict(set)
        org_role_subroles = defaultdict(set)
        location_orgs = defaultdict(set)
        current_org = None
        current_role = None

        def process_entities():
            log("=== PROCESSING ENTITIES ===")
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get("type") != "hypertextlabels":
                        continue

                    value = res.get("value", {})
                    label = value.get("hypertextlabels", [""])[0]
                    text = value.get("text", "").strip()
                    eid = res.get("id")

                    if not all([label, text, eid]):
                        continue

                    entry = {"label": label, "text": text, "uri": None}

                    try:
                        if label == "Organization Name":
                            uri = self.org_name[self._clean_uri(text)]
                            g.add((uri, RDF.type, self.base.Organization))
                            entry["uri"] = uri

                        elif label == "Person Name":
                            uri = self.person_name[self._clean_uri(text)]
                            g.add((uri, RDF.type, self.base.Person))
                            entry["uri"] = uri

                        elif label == "Organization Role":
                            cleaned = self._clean_uri(text)
                            uri = self.org_role[cleaned]
                            g.add((uri, RDFS.subClassOf, self.base.Organization))
                            entry["uri"] = uri

                        elif label == "Organization Sub-Role":
                            cleaned = self._clean_uri(text)
                            uri = self.org_sub_role[cleaned]
                            entry["uri"] = uri

                        elif label == "Person Position":
                            deferred_positions.append((eid, text))
                            entry["deferred"] = True

                        elif label == "Location":
                            uri = self.loc[self._clean_uri(text)]
                            g.add((uri, RDF.type, self.base.Location))
                            entry["uri"] = uri

                        elif label == "Location Type":
                            cleaned = self._clean_uri(text)
                            uri = self.location_type[cleaned]
                            g.add((uri, RDFS.subClassOf, self.base.Location))
                            entry["uri"] = uri

                        registry[eid] = entry

                    except Exception as e:
                        log(f"Error processing entity {eid}: {str(e)}")
                        continue

        process_entities()

        def process_deferred_positions():
            log("\n=== PROCESSING POSITIONS ===")
            for eid, text in deferred_positions:
                cleaned = self._clean_uri(text)
                uri = self.person_position[cleaned]
                g.add((uri, RDFS.subClassOf, self.base.Person))
                registry[eid]["uri"] = uri

        process_deferred_positions()

        def process_relations():
            nonlocal current_org, current_role
            log("\n=== PROCESSING RELATIONS ===")
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get("type") != "relation":
                        continue

                    f, t = res["from_id"], res["to_id"]
                    fe, te = registry.get(f, {}), registry.get(t, {})
                    if not fe or not te or not fe.get("uri") or not te.get("uri"):
                        continue

                    labels = {fe["label"], te["label"]}

                    # Organization ↔ Role
                    if labels == {"Organization Name", "Organization Role"}:
                        org = fe if fe["label"] == "Organization Name" else te
                        role = te if fe["label"] == "Organization Name" else fe
                        org_uri = org["uri"]
                        role_uri = role["uri"]
                        org_roles[org_uri].add(role_uri)
                        current_org = org_uri  # Track the current org-role context
                        current_role = role_uri
                        log(f"Linked {org_uri} to role {role_uri}")

                    # Role ↔ Sub-Role (Only applies to the current org-role context)
                    elif labels == {"Organization Role", "Organization Sub-Role"}:
                        role_entry = fe if fe["label"] == "Organization Role" else te
                        subrole_entry = te if fe["label"] == "Organization Role" else fe
                        role_uri = role_entry["uri"]
                        subrole_uri = subrole_entry["uri"]
                        
                        # Add subrole ONLY to the current org-role pair
                        if current_org and current_role == role_uri:
                            org_role_subroles[(current_org, current_role)].add(subrole_uri)
                            log(f"Added subrole {subrole_uri} to {current_org} {current_role}")


                    # Organization ↔ Location
                    elif labels == {"Organization Name", "Location"}:
                        org_entry = fe if fe["label"] == "Organization Name" else te
                        loc_entry = te if fe["label"] == "Organization Name" else fe
                        org_uri = org_entry["uri"]
                        loc_uri = loc_entry["uri"]
                        location_orgs[loc_uri].add(org_uri)
                        g.add((org_uri, self.rel.hasLocationAt, loc_uri))
                        g.add((loc_uri, self.rel.isLocationOf, org_uri))
                        log(f"Linked {org_uri} to location {loc_uri}")

                    # Location ↔ Type
                    elif labels == {"Location", "Location Type"}:
                        loc_entry = fe if fe["label"] == "Location" else te
                        type_entry = te if fe["label"] == "Location" else fe
                        loc_uri = loc_entry["uri"]
                        type_text = type_entry["text"]
                        type_uri = self.location_type[self._clean_uri(type_text)]
                        g.add((loc_uri, self.isInstanceOf, type_uri))
                        log(f"Linked {loc_uri} to type {type_uri}")

        process_relations()
        def get_local_name(uri: str) -> str:
            """
            Extract the local name from a URI.
            Example: 
                Input: "http://example.org/org_role/syndication_agent"
                Output: "syndication_agent"
            """
            parts = uri.rsplit('/', 1)[-1].rsplit('#', 1)[-1]
            return parts
        def assign_final_roles():
            log("\n=== FINAL ROLE ASSIGNMENTS ===")
            for org_uri, roles in org_roles.items():
                for role_uri in roles:
                   
                    
                    # Add combined roles ONLY if the org has the specific subrole for this role
                    subroles = org_role_subroles.get((org_uri, role_uri), set())  # Key: (org, role)
                    if subroles:
                        for subrole_uri in subroles:
                            base_local = get_local_name(role_uri)
                            sub_local = get_local_name(subrole_uri)
                            combined_role_label = f"{sub_local}_{base_local}"
                            combined_role_uri = self.org_role[combined_role_label]
                            g.add((URIRef(org_uri), self.isInstanceOf, combined_role_uri))
                    else:
                        g.add((URIRef(org_uri), self.isInstanceOf, URIRef(role_uri)))
        assign_final_roles()


        def process_position_relations():
            position_to_org = {}
            person_to_positions = defaultdict(lambda: defaultdict(set))  # {person_uri: {org_uri: set(position_uris)}}

            # First pass: Map Position annotations to their Organizations
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get("type") != "relation":
                        continue
                    from_id, to_id = res["from_id"], res["to_id"]
                    from_ent = registry.get(from_id, {})
                    to_ent = registry.get(to_id, {})
                    
                    # Check if this relation connects a Position to an Organization
                    labels = {from_ent.get("label"), to_ent.get("label")}
                    if labels == {"Organization Name", "Person Position"}:
                        if from_ent.get("label") == "Person Position":
                            position_id, org_id = from_id, to_id
                        else:
                            position_id, org_id = to_id, from_id
                        
                        # Store the organization URI for this position
                        position_to_org[position_id] = registry[org_id]["uri"]

            # Second pass: Map Persons to their Positions and Organizations
            for ann in doc.get("annotations", []):
                for res in ann.get("result", []):
                    if res.get("type") != "relation":
                        continue
                    from_id, to_id = res["from_id"], res["to_id"]
                    from_ent = registry.get(from_id, {})
                    to_ent = registry.get(to_id, {})
                    
                    # Check if this relation connects a Person to a Position
                    labels = {from_ent.get("label"), to_ent.get("label")}
                    if labels == {"Person Name", "Person Position"}:
                        if from_ent.get("label") == "Person Position":
                            position_id, person_id = from_id, to_id
                        else:
                            position_id, person_id = to_id, from_id
                        
                        # Get the organization linked to this position
                        org_uri = position_to_org.get(position_id)
                        if org_uri:
                            person_uri = registry[person_id]["uri"]
                            position_uri = registry[position_id]["uri"]
                            # Track all positions for this person at this organization
                            person_to_positions[person_uri][org_uri].add(position_uri)

            return person_to_positions

        org_position_persons = process_position_relations()

        def get_position_name(pos_uri: str) -> str:
            return next(
                (entry["text"] for entry in registry.values() 
                if entry.get("uri") == pos_uri and entry.get("label") == "Person Position"),
                None  # Fallback if not found
            )

        def get_org_name(org_uri: str) -> str:
            return next(
                (entry["text"] for entry in registry.values() 
                if entry.get("uri") == org_uri and entry.get("label") == "Organization Name"),
                None
            )
        def emit_employment_triples():
            log("\n=== EMITTING EMPLOYMENT TRIPLES ===")

            for person_uri, orgs_dict in org_position_persons.items():
                g.add((URIRef(person_uri), RDF.type, self.base.Person))

                for org_uri, position_uris in orgs_dict.items():
                    g.add((URIRef(org_uri), RDF.type, self.base.Organization))
                    g.add((URIRef(person_uri), self.rel.isEmployedBy, URIRef(org_uri)))
                    g.add((URIRef(org_uri), self.rel.hasEmployee, URIRef(person_uri)))

                    for pos_uri in position_uris:
                        # Emit isInstanceOf triple
                        g.add((URIRef(person_uri), self.isInstanceOf, URIRef(pos_uri)))
                        g.add((URIRef(pos_uri), RDF.type, self.base.Position))  
                        pos_name = get_position_name(pos_uri)
                        if pos_name:
                            g.add((URIRef(pos_uri), self.base.positionName, Literal(pos_name)))

                        # Reified Employment relationship as blank node
                        employment_node = BNode()
                        g.add((URIRef(person_uri), self.rel.holdsPositionAt, employment_node))
                        g.add((employment_node, RDF.type, self.rel.Employment))
                        g.add((employment_node, self.rel.organization, URIRef(org_uri)))
                        g.add((employment_node, self.rel.position, URIRef(pos_uri)))
        emit_employment_triples()
        log(f"Generated {len(g)} triples for employment relationships")

        return g
                

    def _clean_uri(self, text: str) -> str:
        """Enhanced URI cleaning with special character handling."""
        text = text.replace('\\n', '_').replace('\\r', '_')
        text = text.strip().lower()
        replacements = {
            '&': 'and',
            ',': '',
            '/': '_',
            '\n': '_',
            '\r': '_',
            '\t': '_',
            '-': '_',
            ' ': '_'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        cleaned = ''.join(c for c in text if c.isalnum() or c == '_')
        cleaned = re.sub(r'_+', '_', cleaned)
        return cleaned.strip('_')

        
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
    main("/Users/vidhyakshayakannan/Downloads/the_rest_updated 1.json", "./extracted_content")    
    main("/Users/vidhyakshayakannan/Downloads/semi_cleaned_docs.json", "./extracted_content")
    # main("/Users/vidhyakshayakannan/fsil-e3-qa/json_annotations/object_115.json", "./extracted_content_debug")
