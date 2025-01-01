"""
query_generation.py

Changes:
 - Paths come from config
 - Writes queries to queries/ 
 - Loads ontology from extracted_content/ontology.ttl
 - Additional comments
"""

import os
import glob
import re
from rdflib import Graph
from typing import List, Set

import config

class QueryGenerator:
    def __init__(self, ontology_file_path, data_directory):
        self.ontology_graph = self.load_graph(ontology_file_path)
        self.data_files     = glob.glob(os.path.join(data_directory, "*.ttl"))  

    def load_graph(self, ttl_file_path):
        g = Graph()
        g.parse(ttl_file_path, format="ttl")
        return g

    def L1_person(self, data_graph, document_name):
        """
        Generate queries for Person
        """
        query_subclasses = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX person: <http://example.org/person/>
        SELECT ?subclass WHERE {
            ?subclass rdfs:subClassOf person:Person .
        }
        """
        try:
            subclass_results = self.ontology_graph.query(query_subclasses)
            subclasses = {str(row.subclass).split("/")[-1] for row in subclass_results}
        except Exception as e:
            print(f"Error querying subclasses of person:Person: {e}")
            subclasses = set()

        query_nlq = """
        PREFIX person: <http://example.org/person/>
        PREFIX rel:   <http://example.org/relation/>
        SELECT ?name ?position ?organization WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?organization .
            BIND(REPLACE(STR(?person), ".*/", "") AS ?name)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        try:
            results = data_graph.query(query_nlq)
        except Exception as e:
            print(f"Error executing query on data graph for {document_name}: {e}")
            return []

        nlqs = []
        for row in results:
            try:
                name, position, org = row
                position_name = position.split("/")[-1]
                if position_name in subclasses:
                    nlqs.append(f"What is the position of {name} in {org}?")
                    nlqs.append(f"Who is the {position_name.lower()} of {org}?")
                    nlqs.append(f"Where does {name} work as {position_name.lower()}?")
            except Exception as e:
                print(f"Error processing row {row}: {e}")

        cleaned = [self.clean_query(q) for q in nlqs]
        self.save_queries(cleaned, document_name, append=True, prefix="L1")
        return nlqs

    def Level_1_location(self, data_graph, document_name):
        """
        Generate queries for Location
        """
        nlqs = []
        location_query = """
        PREFIX loc: <http://example.org/location/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?location ?organization WHERE {
            ?loc a loc:Location ;
                 rel:isLocationOf ?organization .
            BIND(REPLACE(STR(?loc), ".*/", "") AS ?location)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        results = data_graph.query(location_query)
        for row in results:
            location = row[0].split("/")[-1]
            org = row[1].split("/")[-1].replace("%5Cn", "").replace("%2C", ",")
            nlqs.append(f"Where is {org} located?")
            nlqs.append(f"Which organization is located at {location}?")
        
        cleaned = [self.clean_query(q) for q in set(nlqs)]
        self.save_queries(cleaned, document_name, append=True, prefix="L1")

    def Level_1_Roles(self, data_graph, document_name):
        """
        Generate queries for Organization roles
        """
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX org: <http://example.org/organization/>
        SELECT ?role ?subRole WHERE {
            ?role rdfs:subClassOf org:Organization .
            OPTIONAL { ?subRole rdfs:subClassOf ?role . }
        }
        """
        results = self.ontology_graph.query(query)
        roles_and_sub_roles = {}
        for row in results:
            role = str(row.role).split("/")[-1]
            sub_role = str(row.subRole).split("/")[-1] if row.subRole else None
            if role not in roles_and_sub_roles:
                roles_and_sub_roles[role] = set()
            if sub_role:
                roles_and_sub_roles[role].add(sub_role)

        nlqs = []
        org_query = """
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?organization ?role ?employee WHERE {
            ?org a org:Organization ;
                 a ?role ;
                 rel:hasEmployee ?employee .
            FILTER (?role != org:Organization)
            BIND(REPLACE(STR(?org), ".*/", "") AS ?organization)
            BIND(REPLACE(STR(?role), ".*/", "") AS ?role)
            BIND(REPLACE(STR(?employee), ".*/", "") AS ?employee)
        }
        """
        results = data_graph.query(org_query)
        for row in results:
            org, role, employee = row
            role_name = role.split("/")[-1]
            sub_roles = roles_and_sub_roles.get(role_name, set())
            nlqs.append(f"What is the role of {org}?")
            for sr in sub_roles:
                nlqs.append(f"Who is the {sr.lower()} {role_name.lower()} in the agreement?")

        cleaned = [self.clean_query(q) for q in set(nlqs)]
        self.save_queries(cleaned, document_name, append=True, prefix="L1")

    def L2_person_org(self, data_graph, document_name):
        """
        Generate queries for advanced Person-Organization relationships
        """
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX org: <http://example.org/organization/>
        SELECT ?role ?subRole WHERE {
            ?role rdfs:subClassOf org:Organization .
            OPTIONAL { ?subRole rdfs:subClassOf ?role . }
        }
        """
        results = self.ontology_graph.query(query)
        combined_roles = {}
        for row in results:
            role = str(row.role).split("/")[-1]
            sub_role = str(row.subRole).split("/")[-1] if row.subRole else None
            combined_roles[role] = sub_role

        nlqs = []
        person_org_query = """
        PREFIX person: <http://example.org/person/>
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?person ?position ?organization ?role WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?org .
            ?org a org:Organization ;
                 a ?role .
            FILTER(?position != person:Person && ?role != org:Organization)
        }
        """
        results = data_graph.query(person_org_query)
        for row in results:
            person, position, organization, role = row
            person_name = str(person).split("/")[-1]
            position_name = str(position).split("/")[-1]
            organization_name = str(organization).split("/")[-1]
            role_name = str(role).split("/")[-1]
            sub_role = combined_roles.get(role_name, None)
            role_with_sub_role = f"{sub_role} {role_name}" if sub_role else role_name

            nlqs.append(f"Who is the {position_name.lower()} of the organization that is the {role_with_sub_role.lower()}?")
            nlqs.append(f"{person_name} holds what position in the organization that is the {role_with_sub_role.lower()}?")
            nlqs.append(f"What is the role of the organization in which {person_name} is {position_name.lower()}?")

        cleaned = [self.clean_query(q) for q in nlqs]
        self.save_queries(cleaned, document_name, append=True, prefix="L2")

    def save_queries(self, queries, document_name, append=False, prefix="L1"):
        os.makedirs(config.QUERIES_DIR, exist_ok=True)
        file_path = os.path.join(config.QUERIES_DIR, f"{prefix}_queries_{document_name}.txt")
        mode = 'a' if append else 'w'
        with open(file_path, mode) as f:
            for q in queries:
                f.write(q + "\n")

    def clean_query(self, query):
        cleaned_query = re.sub(r'[\d]+Cn|[\d]+C|C[0-9]+|A[0-9]+|5Cn|C2|A0+26', '', query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'[%\\]+', ' ', cleaned_query)
        cleaned_query = re.sub(r'[_]+', ' ', cleaned_query)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
        return cleaned_query.strip()

def main():
    """
    Example usage if run standalone.
    """
    ontology_path = os.path.join(config.EXTRACTED_CONTENT_DIR, "ontology.ttl")
    qg = QueryGenerator(ontology_file_path=ontology_path, data_directory=config.EXTRACTED_CONTENT_DIR)
    for data_file_path in qg.data_files:
        data_graph = qg.load_graph(data_file_path)
        doc_name = os.path.basename(data_file_path).replace('.ttl', '')
        
        qg.L1_person(data_graph, doc_name)
        qg.Level_1_location(data_graph, doc_name)
        qg.Level_1_Roles(data_graph, doc_name)
        qg.L2_person_org(data_graph, doc_name)

if __name__ == "__main__":
    main()
