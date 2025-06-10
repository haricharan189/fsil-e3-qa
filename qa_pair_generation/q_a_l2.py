import pandas as pd
from rdflib import Graph, Namespace, URIRef
from urllib.parse import unquote
import glob
import os
import re
from typing import Dict, Set
import itertools
from itertools import combinations
from collections import defaultdict

class QuestionGenerator:
    def __init__(self, ttl_dir="extracted_content"):
        self.ttl_dir = ttl_dir
        self.person_name = Namespace("http://example.org/person_name/")
        self.person_position = Namespace("http://example.org/person_position/")
        self.org_name = Namespace("http://example.org/org_name/")
        self.rel = Namespace("http://example.org/relation/")
        self.loc = Namespace("http://example.org/location/")
        self.location_type = Namespace("http://example.org/location_type/")
        self.isInstanceOf = URIRef("http://example.org/isInstanceOf/")
        self.org_role = Namespace("http://example.org/org_role/")
        
    def load_graph(self, ttl_file):
        """Load a TTL file into an RDFlib Graph"""
        g = Graph()
        g.parse(ttl_file, format="turtle")
        return g
    
    def clean_uri(self, uri):
        """Clean URI to get readable text"""
        text = str(uri).split('/')[-1]
        text = text.replace('%20', ' ').replace('%2C', ',').replace('%5Cn', '')
        text = text.replace('_', ' ').replace('%C2%A0', ' ')
        return text.strip()
    
    def get_person_positions(self, graph):
        """Get all persons and their positions from the graph"""
        person_positions = {}
        query = """
        SELECT DISTINCT ?person (GROUP_CONCAT(?position; separator="|") as ?positions)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                   <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        GROUP BY ?person
        """     
        results = graph.query(query)
        
        for row in results:
            person_uri = row[0]
            position_uris = row[1].split('|')
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris) 
            person_positions[person_name] = positions
            
        return person_positions
    
    def get_org_people_positions(self, graph):
        """Get all organizations and their people with positions"""
        org_people_positions = {}
        query = """
        SELECT DISTINCT ?org ?person (GROUP_CONCAT(?position; separator="|") as ?positions)
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                   <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        GROUP BY ?org ?person
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            person_uri = row[1]
            position_uris = row[2].split('|')
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris)
            if org_name not in org_people_positions:
                org_people_positions[org_name] = {}
            org_people_positions[org_name][person_name] = positions
            
        return org_people_positions
    
    def get_org_roles(self, graph):
        """Get all organizations and their roles from the graph, resolving subroles to roles"""
        org_roles = {}

        # SPARQL query to get all organizations and their roles, resolving subroles
        query = """
                SELECT DISTINCT ?org (GROUP_CONCAT(DISTINCT ?finalRole; separator="|") as ?roles)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?roleOrSubrole .

            OPTIONAL {
                ?roleOrSubrole rdfs:subClassOf ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }

            # Use COALESCE to handle unbound ?role values
            BIND(COALESCE(?role, ?roleOrSubrole) AS ?finalRole)
            FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/"))
        }
        GROUP BY ?org

        """

        results = graph.query(query)

        for row in results:
            org_uri = row[0]
            role_uris = row[1].split('|')

            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            roles = set(self.clean_uri(role) for role in role_uris)

            org_roles[org_name] = roles

        return org_roles
    
    def get_location_type(self, graph):
        """
        Return a dict mapping each location to the set of its types.
        { location_name: set(location_type_names) }
        """
        location_types = {}

        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel:  <http://example.org/relation/>

        SELECT DISTINCT ?loc (GROUP_CONCAT(DISTINCT ?type; separator="|") AS ?types)
        WHERE {
            ?loc a base:Location ;
                <http://example.org/isInstanceOf/> ?type ;  # Correct predicate
                rel:isLocationOf ?org .                     # Correct relationship direction
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?loc
        """

        results = graph.query(query)
        for loc_uri, types_concat in results:
            loc_name = self.clean_uri(loc_uri)
            type_uris = types_concat.split("|") if types_concat else []
            types = {self.clean_uri(t) for t in type_uris}
            location_types[loc_name] = types

        return location_types



    def get_location_org_people_positions(self, graph):
        """Get all locations, their organizations, and people with positions (using reified employment relationships)."""
        location_org_people = {}
        
        # Updated SPARQL query to handle reified employment relationships
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?loc ?org ?person ?position
        WHERE {
            ?loc a base:Location .
            ?org rel:hasLocationAt ?loc .
            ?person rel:isEmployedBy ?org ;
                rel:holdsPositionAt ?employment .
            ?employment rel:position ?position ;
                        rel:organization ?org .
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row.loc
            org_uri = row.org
            person_uri = row.person
            position_uri = row.position
            
            # Clean URIs to human-readable format
            location = self.clean_uri(loc_uri)
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Build nested dictionary structure
            location_entry = location_org_people.setdefault(location, {})
            org_entry = location_entry.setdefault(org_name, {})
            person_entry = org_entry.setdefault(person_name, set())
            person_entry.add(position)
            
        return location_org_people
    
    def get_org_employee_positions(self, graph):
        """Get all organizations with their employees and positions using reified relationships"""
        org_info = {}

        # SPARQL query for reified employment structure
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?org ?employee ?position
        WHERE {
            ?org a base:Organization .
            ?employee rel:holdsPositionAt ?employment .
            ?employment rel:organization ?org ;
                        rel:position ?position .
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row.org
            employee_uri = row.employee
            position_uri = row.position
            
            # Clean URIs with proper type handling
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri)
            position = self.clean_uri(position_uri)

            # Initialize organization entry
            if org_name not in org_info:
                org_info[org_name] = {
                    'employees': set(),
                    'positions': defaultdict(set)
                }

            # Add employee to organization
            org_info[org_name]['employees'].add(employee_name)
            
            # Add position with employee as holder
            org_info[org_name]['positions'][position].add(employee_name)
        
        return org_info
    
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations and their employees with roles"""
        org_info = {}

        # Simplified SPARQL query without subrole handling
        query = """
        PREFIX base: <http://example.org/base/>
        SELECT DISTINCT ?org ?employee ?role
        WHERE {
            ?org a base:Organization ;
                <http://example.org/isInstanceOf/> ?role .
            OPTIONAL {
                ?org <http://example.org/relation/hasEmployee> ?employee .
            }
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
        }
        """

        results = graph.query(query)
        
        for row in results:
            org_uri = row.org
            employee_uri = row.employee if row.employee else None
            role_uri = row.role

            # Convert URIs to readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri) if employee_uri else None
            role = self.clean_uri(role_uri)

            # Initialize org entry if not present
            if org_name not in org_info:
                org_info[org_name] = {'employees': set(), 'roles': set()}

            # Add role and employee
            org_info[org_name]['roles'].add(role)
            if employee_name:
                org_info[org_name]['employees'].add(employee_name)

        return org_info
    
    def get_location_org_roles(self, graph: Graph) -> Dict[str, Dict[str, Set[str]]]:
        """Get location -> organization -> merged roles with subroles"""
        location_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?org ?role ?subrole ?location
        WHERE {
            ?org a base:Organization ;
                <http://example.org/isInstanceOf/> ?role ;
                rel:hasLocationAt ?location .
            
            OPTIONAL {
                ?org <http://example.org/isInstanceOf/> ?subrole .
                ?subrole rdfs:subClassOf ?role .
            }
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org = self.clean_uri(row.org)
            role = self.clean_uri(row.role)
            subrole = self.clean_uri(row.subrole) if row.subrole else None
            raw_location = unquote(str(row.location))
            location = self.clean_uri(raw_location)
            if subrole:
                merged_role = f"{subrole} {role}"
                location_map[location][org].add(merged_role)
            else:
                location_map[location][org].add(role)
        
        return location_map
    
    def get_location_org_data(self, graph):
            """Retrieve organizations, their associated locations, and location types."""
            location_org_data = {}
            query = """
                SELECT DISTINCT ?org ?loc (GROUP_CONCAT(?type; separator="|") as ?types)
                WHERE {
                    ?org a <http://example.org/base/Organization> ;
                        <http://example.org/relation/hasLocationAt> ?loc .
                    ?loc a <http://example.org/base/Location> ;
                        <http://example.org/isInstanceOf/> ?type .
                    FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
                }
                GROUP BY ?org ?loc
            """
            
            # Execute the query and get results
            results = graph.query(query)
            print("Query executed. Number of results:", len(results))  # Debugging

            # Process query results
            for row in results:
                print("Processing row: ", row)  # Debugging
                org_uri = row[0]
                location_uri = row[1]
                location_type_uris = row[2].split('|')  # Split the types into a list

                # Clean the URIs to get readable text
                org_name = self.clean_uri(org_uri)
                location = self.clean_uri(location_uri)
                location_types = [self.clean_uri(type_uri) for type_uri in location_type_uris]

                # Debugging cleaned URIs and types
                print(f"Cleaned URIs: Organization: {org_name}, Location: {location}, Location Types: {location_types}")  # Debugging

                # Initialize location-org data structure if not already initialized
                if location not in location_org_data:
                    location_org_data[location] = {}

                # Store the organization and its associated location types
                if org_name not in location_org_data[location]:
                    location_org_data[location][org_name] = {'location_types': set(location_types)}

                # Add location types to the organization entry
                location_org_data[location][org_name]['location_types'].update(location_types)
                
                # Debugging state of location-org data after processing each row
                print(f"Updated location-org data: {location_org_data}")  # Debugging

            return location_org_data
   
    

    
    def generate_person_position_questions_plural(self, graph, doc_num):
        """What are the positions of [Person Name]?"""
        questions = []
        query = """
        SELECT DISTINCT ?person (GROUP_CONCAT(?position; separator="|") as ?positions)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        GROUP BY ?person
        HAVING (COUNT(?position) > 1)
        """
        
        results = graph.query(query)
        
        for row in results:
            person_uri = row[0]
            position_uris = row[1].split('|')
            person_name = self.clean_uri(person_uri)
            positions = [self.clean_uri(pos) for pos in position_uris]
            
            questions.append({
                'question': f"What are the positions of {person_name}?",
                'answer': ", ".join(positions),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        
        return questions
    
    def generate_organization_representative_questions_plural(self, graph, doc_num):
        """Who are the representatives of [Organization Name]?"""
        questions = []
        query = """
        SELECT ?org (GROUP_CONCAT(?person; separator="|") as ?reps)
        WHERE {
            ?person <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?org
        HAVING (COUNT(?person) > 1)
        """
        
        results = graph.query(query)
        
        for row in results:
            org = self.clean_uri(row[0])
            reps = [self.clean_uri(p) for p in row[1].split('|')]
            
            questions.append({
                'question': f"Who are the representatives of {org}?",
                'answer': ", ".join(reps),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        
        return questions
    
    def generate_org_role_questions_plural(self, graph, doc_num):
        """What are the roles of [Organization Name] in the agreement?"""
        questions = []
        role_subrole_map = self.get_org_roles(graph)
    
        query = """
        SELECT ?org (GROUP_CONCAT(?role; separator="|") as ?roles)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?role .
            { FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/")) }
            UNION
            { FILTER(STRSTARTS(STR(?role), "http://example.org/org_sub_role/")) }
        }
        GROUP BY ?org
        HAVING (COUNT(?role) > 1)
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            role_uris = row[1].split('|')
            
            org_name = self.clean_uri(org_uri)
            roles = [self.clean_uri(role) for role in role_uris]
            processed_roles = []
            for role in roles:
                if role in role_subrole_map:
                    processed_roles.append(f"{role} ({' '.join(role_subrole_map[role])})")
                else:
                    processed_roles.append(role)
            
            answer = ", ".join(sorted(processed_roles))
            
            questions.append({
                'question': f"What are the roles of {org_name} in the agreement?",
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        
        return questions
    
    
    def get_org_role_subrole_map(self, graph):
        """
        Build a mapping from each base org_role URI to a set of its sub_role URIs,
        using rdfs:subClassOf relationships from the RDF graph.
        """
        from rdflib.namespace import RDFS

        role_subrole_map = {}

        print("▶ Running get_org_role_subrole_map…")

        # SPARQL query to get sub-role → role mappings
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?role ?sub_role
        WHERE {
            ?sub_role rdfs:subClassOf ?role .
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
        }
        """

        results = list(graph.query(query))
        print(f"  ► SPARQL returned {len(results)} role→subrole pairs")

        for role_uri, sub_role_uri in results:
            role_uri_str = str(role_uri)
            sub_role_uri_str = str(sub_role_uri)

            print(f"    – Found: role={role_uri_str}, subrole={sub_role_uri_str}")

            if role_uri_str not in role_subrole_map:
                role_subrole_map[role_uri_str] = set()

            role_subrole_map[role_uri_str].add(sub_role_uri_str)

        print("  ▶ Built role_subrole_map:")
        for role, subs in role_subrole_map.items():
            print(f"    • {role} → {sorted(subs)}")

        return role_subrole_map


    def get_org_roles_with_subroles(self, graph):
        """Return organization roles, using 'subrole (ParentRole)' format if subroles exist."""

        # 1. subrole -> parent role mapping
        subrole_to_parent = {}
        query1 = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?sub_role ?parent_role
        WHERE {
            ?sub_role rdfs:subClassOf ?parent_role .
            FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
            FILTER(STRSTARTS(STR(?parent_role), "http://example.org/org_role/"))
        }
        """
        for sub_role_uri, parent_role_uri in graph.query(query1):
            sub = self.clean_uri(sub_role_uri)
            parent = self.clean_uri(parent_role_uri)
            subrole_to_parent[sub] = parent

        # 2. Collect all role URIs per organization
        org_to_roles = defaultdict(set)

        query2 = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel:  <http://example.org/isInstanceOf/>
        SELECT ?org ?role
        WHERE {
            ?org a base:Organization ;
                rel: ?role .
            FILTER(
                STRSTARTS(STR(?role), "http://example.org/org_role/") ||
                STRSTARTS(STR(?role), "http://example.org/org_sub_role/")
            )
        }
        """
        for org_uri, role_uri in graph.query(query2):
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            org_to_roles[org_name].add(role)

        # 3. Normalize: Prefer subrole (ParentRole), skip base role if overridden
        org_roles = defaultdict(lambda: {'roles': set()})
        for org, roles in org_to_roles.items():
            parents_used = set()

            # First add all subroles with parents
            for role in roles:
                if role in subrole_to_parent:
                    parent = subrole_to_parent[role]
                    org_roles[org]['roles'].add(f"{role} {parent}")
                    parents_used.add(parent)

            # Then add remaining roles that were not overridden by subroles
            for role in roles:
                if role not in subrole_to_parent and role not in parents_used:
                    org_roles[org]['roles'].add(role)

        return org_roles

    
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations, their employees, and cleanly deduplicated roles (subrole if available)."""
        from collections import defaultdict

        org_info = defaultdict(lambda: {'employees': set(), 'roles': set()})
        subrole_to_parent = {}

        # First: build subrole → parent map
        query_hierarchy = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?sub_role ?parent_role
        WHERE {
            ?sub_role rdfs:subClassOf ?parent_role .
            FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
            FILTER(STRSTARTS(STR(?parent_role), "http://example.org/org_role/"))
        }
        """
        for sub_role_uri, parent_role_uri in graph.query(query_hierarchy):
            sub = self.clean_uri(sub_role_uri)
            parent = self.clean_uri(parent_role_uri)
            subrole_to_parent[sub] = parent

        # Second: query all org → role + employee links
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel:  <http://example.org/isInstanceOf/>
        PREFIX reln: <http://example.org/relation/>

        SELECT DISTINCT ?org ?employee ?role
        WHERE {
            ?org a base:Organization ;
                rel: ?role .
            OPTIONAL { ?org reln:hasEmployee ?employee . }
            FILTER(
                STRSTARTS(STR(?role), "http://example.org/org_role/") ||
                STRSTARTS(STR(?role), "http://example.org/org_sub_role/")
            )
        }
        """
        for org_uri, emp_uri, role_uri in graph.query(query):
            org_name = self.clean_uri(org_uri)
            role_key = self.clean_uri(role_uri)
            employee = self.clean_uri(emp_uri) if emp_uri else None

            if role_key in subrole_to_parent:
                display_role = f"{role_key} ({subrole_to_parent[role_key]})"
            else:
                display_role = role_key

            org_info[org_name]['roles'].add(display_role)
            if employee:
                org_info[org_name]['employees'].add(employee)

        return org_info

    
    def _get_org_locations(self, graph):
        """Helper to fetch organization locations with proper SPARQL syntax"""
        org_locations = defaultdict(list)
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?loc; separator="|") AS ?locs)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                 <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?org
        """
        results = graph.query(query)
        for org_uri, locs_str in results:
            org_name = self.clean_uri(org_uri)
            locations = [self.clean_uri(l) for l in locs_str.split('|') if l]
            org_locations[org_name] = locations
        return org_locations
    
    def get_person_positions(self, graph):
        """Get all persons and their positions from the graph, including positions via employment relationships."""
        person_positions = {}
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel:  <http://example.org/relation/>
        
        SELECT DISTINCT ?person (GROUP_CONCAT(DISTINCT ?position; separator="|") AS ?positions)
        WHERE {
            {
                # Direct positions via isInstanceOf
                ?person a base:Person ;
                        <http://example.org/isInstanceOf/> ?position .
                FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
            }
            UNION
            {
                # Positions via employment relationships
                ?person a base:Person ;
                        rel:holdsPositionAt [ rel:position ?position ] .
                FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
            }
        }
        GROUP BY ?person
        """
        results = graph.query(query)
        
        for row in results:
            person_uri, positions_str = row
            if positions_str:
                position_uris = positions_str.split("|")
                person_name = self.clean_uri(person_uri)
                positions = {self.clean_uri(pos) for pos in position_uris}
                person_positions[person_name] = positions
                
        return person_positions
        
    def get_org_people_positions(self, graph):
        """Get all organizations and their people with positions"""
        org_people_positions = {}
        query = """
        SELECT DISTINCT ?org ?person (GROUP_CONCAT(?position; separator="|") as ?positions)
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                   <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        GROUP BY ?org ?person
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            person_uri = row[1]
            position_uris = row[2].split('|')
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris)
            if org_name not in org_people_positions:
                org_people_positions[org_name] = {}
            org_people_positions[org_name][person_name] = positions
            
        return org_people_positions
    
    def get_org_roles(self, graph):
        """Get all organizations and their roles from the graph, resolving subroles to roles"""
        org_roles = {}

        # SPARQL query to get all organizations and their roles, resolving subroles
        query = """
                SELECT DISTINCT ?org (GROUP_CONCAT(DISTINCT ?finalRole; separator="|") as ?roles)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?roleOrSubrole .

            OPTIONAL {
                ?roleOrSubrole rdfs:subClassOf ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }

            # Use COALESCE to handle unbound ?role values
            BIND(COALESCE(?role, ?roleOrSubrole) AS ?finalRole)
            FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/"))
        }
        GROUP BY ?org

        """

        results = graph.query(query)

        for row in results:
            org_uri = row[0]
            role_uris = row[1].split('|')

            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            roles = set(self.clean_uri(role) for role in role_uris)

            org_roles[org_name] = roles

        return org_roles


    def get_location_org_people_positions(self, graph):
        """Get all locations, their organizations, and people with positions (using reified employment relationships)."""
        location_org_people = {}
        
        # Updated SPARQL query to handle reified employment relationships
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?loc ?org ?person ?position
        WHERE {
            ?loc a base:Location .
            ?org rel:hasLocationAt ?loc .
            ?person rel:isEmployedBy ?org ;
                rel:holdsPositionAt ?employment .
            ?employment rel:position ?position ;
                        rel:organization ?org .
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row.loc
            org_uri = row.org
            person_uri = row.person
            position_uri = row.position
            
            # Clean URIs to human-readable format
            location = self.clean_uri(loc_uri)
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Build nested dictionary structure
            location_entry = location_org_people.setdefault(location, {})
            org_entry = location_entry.setdefault(org_name, {})
            person_entry = org_entry.setdefault(person_name, set())
            person_entry.add(position)
            
        return location_org_people
    
    def get_org_employee_positions(self, graph):
        """Get all organizations with their employees and positions using reified relationships"""
        org_info = {}

        # SPARQL query for reified employment structure
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?org ?employee ?position
        WHERE {
            ?org a base:Organization .
            ?employee rel:holdsPositionAt ?employment .
            ?employment rel:organization ?org ;
                        rel:position ?position .
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row.org
            employee_uri = row.employee
            position_uri = row.position
            
            # Clean URIs with proper type handling
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri)
            position = self.clean_uri(position_uri)

            # Initialize organization entry
            if org_name not in org_info:
                org_info[org_name] = {
                    'employees': set(),
                    'positions': defaultdict(set)
                }

            # Add employee to organization
            org_info[org_name]['employees'].add(employee_name)
            
            # Add position with employee as holder
            org_info[org_name]['positions'][position].add(employee_name)
        
        return org_info
    
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations and their employees with roles"""
        org_info = {}

        # Simplified SPARQL query without subrole handling
        query = """
        PREFIX base: <http://example.org/base/>
        SELECT DISTINCT ?org ?employee ?role
        WHERE {
            ?org a base:Organization ;
                <http://example.org/isInstanceOf/> ?role .
            OPTIONAL {
                ?org <http://example.org/relation/hasEmployee> ?employee .
            }
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
        }
        """

        results = graph.query(query)
        
        for row in results:
            org_uri = row.org
            employee_uri = row.employee if row.employee else None
            role_uri = row.role

            # Convert URIs to readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri) if employee_uri else None
            role = self.clean_uri(role_uri)

            # Initialize org entry if not present
            if org_name not in org_info:
                org_info[org_name] = {'employees': set(), 'roles': set()}

            # Add role and employee
            org_info[org_name]['roles'].add(role)
            if employee_name:
                org_info[org_name]['employees'].add(employee_name)

        return org_info
    
    def get_location_org_roles(self, graph: Graph) -> Dict[str, Dict[str, Set[str]]]:
        """Get location -> organization -> merged roles with subroles"""
        location_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?org ?role ?subrole ?location
        WHERE {
            ?org a base:Organization ;
                <http://example.org/isInstanceOf/> ?role ;
                rel:hasLocationAt ?location .
            
            OPTIONAL {
                ?org <http://example.org/isInstanceOf/> ?subrole .
                ?subrole rdfs:subClassOf ?role .
            }
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org = self.clean_uri(row.org)
            role = self.clean_uri(row.role)
            subrole = self.clean_uri(row.subrole) if row.subrole else None
            raw_location = unquote(str(row.location))
            location = self.clean_uri(raw_location)
            if subrole:
                merged_role = f"{subrole} {role}"
                location_map[location][org].add(merged_role)
            else:
                location_map[location][org].add(role)
        
        return location_map
    
    def get_location_org_data(self, graph):
            """Retrieve organizations, their associated locations, and location types."""
            location_org_data = {}
            query = """
                SELECT DISTINCT ?org ?loc (GROUP_CONCAT(?type; separator="|") as ?types)
                WHERE {
                    ?org a <http://example.org/base/Organization> ;
                        <http://example.org/relation/hasLocationAt> ?loc .
                    ?loc a <http://example.org/base/Location> ;
                        <http://example.org/isInstanceOf/> ?type .
                    FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
                }
                GROUP BY ?org ?loc
            """
            
            # Execute the query and get results
            results = graph.query(query)
            print("Query executed. Number of results:", len(results))  # Debugging

            # Process query results
            for row in results:
                print("Processing row: ", row)  # Debugging
                org_uri = row[0]
                location_uri = row[1]
                location_type_uris = row[2].split('|')  # Split the types into a list

                # Clean the URIs to get readable text
                org_name = self.clean_uri(org_uri)
                location = self.clean_uri(location_uri)
                location_types = [self.clean_uri(type_uri) for type_uri in location_type_uris]

                # Debugging cleaned URIs and types
                print(f"Cleaned URIs: Organization: {org_name}, Location: {location}, Location Types: {location_types}")  # Debugging

                # Initialize location-org data structure if not already initialized
                if location not in location_org_data:
                    location_org_data[location] = {}

                # Store the organization and its associated location types
                if org_name not in location_org_data[location]:
                    location_org_data[location][org_name] = {'location_types': set(location_types)}

                # Add location types to the organization entry
                location_org_data[location][org_name]['location_types'].update(location_types)
                
                # Debugging state of location-org data after processing each row
                print(f"Updated location-org data: {location_org_data}")  # Debugging

            return location_org_data
    
    def generate_company_by_role_questions_plural(self, graph, doc_num):
        """What companies are the [Organization Role [+Sub-Role] in the agreement?"""
        questions = []
        
        # Get orgs grouped by normalized roles (from existing method)
        org_roles = self.get_org_roles_with_subroles(graph)
        
        # Invert to get role -> organizations mapping
        role_org_map = defaultdict(set)
        for org, data in org_roles.items():
            for role in data['roles']:
                role_org_map[role].add(org)
        
        # Generate questions for roles with multiple organizations
        for role, orgs in role_org_map.items():
            if len(orgs) > 1:
                questions.append({
                    'question': f"What companies are the {role} in the agreement?",
                    'answer': ", ".join(sorted(orgs)),
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })
        
        return questions
    
    
    def generate_org_location_questions_plural(self, graph, doc_num):
        """What are the locations of [Organization Name]?"""
        questions = []
        org_locations = self._get_org_locations(graph)
        
        for org, locations in org_locations.items():
            if len(locations) > 1:
                questions.append({
                    'question': f"What are the locations of {org}?",
                    'answer': ", ".join(sorted(locations)),
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })
        
        return questions

    def generate_location_type_questions_plural(self, graph, doc_num):
        """What type(s) of location is [Location] (e.g., Domestic Lending Office, Eurodollar Lending Office, etc.)?"""
        questions = []
        loc_types = self.get_location_type(graph)  # Ensure this method resolves URIs to readable names
        
        for loc, types in loc_types.items():
            if len(types)>1:
                clean_loc = self.clean_uri(loc)
                answer = ", ".join(sorted(types))
                questions.append({
                    'question': f"What types of location is {clean_loc}?",
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1 
            })
    
        return questions   
    
    def generate_person_organization_questions_plural(self, graph, doc_num):
        """In what organizations does [Person Name] work?"""
        questions = []
        query = """
        SELECT ?person (GROUP_CONCAT(?org; separator="|") as ?orgs)
            WHERE {
                ?person a <http://example.org/base/Person> ;
                    <http://example.org/relation/isEmployedBy> ?org .
            }
            GROUP BY ?person
            HAVING (COUNT(?org) > 1)
        """
        
        results = graph.query(query)
        
        for row in results:
            person = self.clean_uri(row[0])
            orgs = [self.clean_uri(o) for o in row[1].split('|')]
            
            questions.append({
                'question': f"In what organizations does {person} work?",
                'answer': ", ".join(orgs),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        
        return questions

    
    def generate_shared_position_questions(self, graph, doc_num):
            """What is the position held by both [Person Name 1] and [Person Name 2]? (if one)"""
            questions = []
            person_positions = self.get_person_positions(graph)
            if len(person_positions) > 1:
                # Generate all unique combinations of two people
                for person1, person2 in combinations(person_positions.keys(), 2):
                    positions1 = person_positions[person1]
                    positions2 = person_positions[person2]
                    shared_positions = positions1 & positions2
                    
                    if len(shared_positions)==1:
                        answer = ", ".join(sorted(shared_positions))  
                        question = f"What is the position held by both {person1} and {person2}?"
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 1,  
                            'num_set_operations': 1,  
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        })
            
            return questions
    
            

    def generate_shared_role_questions(self, graph, doc_num):
        """"What role do both [Organization 1] and [Organization 2] have in the agreement? [if one]"""
        questions = []
        org_roles = self.get_org_roles(graph)
        print(f"Retrieved org roles: {org_roles}")  
        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]
            
            print(f"\nComparing organizations: {org1} vs {org2}")
            print(f"Roles for {org1}: {roles1}")
            print(f"Roles for {org2}: {roles2}")
            if len(roles1) > 1 and len(roles2) > 1:
                shared_roles = roles1 & roles2
                print(f"Shared roles: {shared_roles}")  
                if len(shared_roles) == 1:
                    answer = next(iter(shared_roles))  
                    print(f"Selected answer: {answer}") 
                    question = f"What role do both {org1} and {org2} have in the agreement?"
                    print(f"Generated question: {question}")  
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,  
                        'num_set_operations': 1,  
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })
        print("\nFinal generated questions:")
        for q in questions:
            print(f"Q: {q['question']} | A: {q['answer']}")

        return questions
    
    def generate_shared_role_subrole_questions(self, graph, doc_num):
        """What company is both the [Org Role (+Sub-Role) 1] and [Org Role (+Sub-Role) 2] in the agreement? [if one]"""
        questions = []
        org_roles = self.get_org_roles(graph)  
        
        for org, roles in org_roles.items():
            if len(roles) > 1:
                # Generate all unique pairs of roles
                for role1, role2 in combinations(roles, 2):
                    clean_role1 = self.clean_uri(role1)
                    clean_role2 = self.clean_uri(role2)
                    
                    question = f"What company is both the {clean_role1} and {clean_role2} in the agreement?"
                    
                    questions.append({
                        'question': question,
                        'answer': org, 
                        'num_hops': 1,
                        'num_set_operations': 1,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })
        
        return questions
    
    def generate_position_organization_questions(self, graph, doc_num):
        """"Who is the [Position] of [Organization]?"""
        questions = {}
        org_info = self.get_org_employee_positions(graph)
        for org_name, info in org_info.items():
            for position, holders in info['positions'].items():
                if len(holders) == 1:  
                    question = f"Who is the {position} of {org_name}?"
                    answer = list(holders)[0] 
                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0  
                    }

        return list(questions.values())
    
    def generate_person_company_role_question(self, graph, doc_num):
            """What is the role in the agreement of the company where [Person Name] is employed?"""
            questions = {}
            org_info = self.get_org_employees_and_roles(graph)

            for org_name, info in org_info.items():
                roles = info['roles'] 
                if len(roles) == 1: 
                    single_role = list(roles)[0]
                    for employee in info['employees']:
                        question = f"What is the role in the agreement of the company where {employee} is employed?"
                        answer = single_role  

                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 2,  
                            'num_set_operations': 0,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0  
                        }

            return list(questions.values())
    
   
    
    def generate_location_role_questions(self, graph, doc_num):
        """What is the role in the agreement of the company associated with [Location]?"""
        questions = {}
        location_data = self.get_location_org_roles(graph)
        
        for location, orgs in location_data.items():
            all_roles = set()
            for roles in orgs.values():
                all_roles.update(roles)
        
            if len(all_roles) == 1:
                role = next(iter(all_roles))
                question = f"What is the role in the agreement of the company associated with {location}?"
                
                questions[question] = {
                    'question': question,
                    'answer': role,
                    'num_hops': 2,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                }
        
        return list(questions.values())
    


    def generate_location_office_questions(self, graph, doc_num):
        """What is the [Location Type] office of [Org Name]?"""
        questions = {}
        location_org_data = self.get_location_org_data(graph)
        for location, orgs in location_org_data.items(): 
            for org_name, location_types in orgs.items(): 
                if isinstance(location_types, dict):
                    location_types = location_types.get('location_types', set())  
                if isinstance(location_types, set):
                    location_types = list(location_types)  
                for location_type in location_types:
                    location_type_cleaned = location_type.strip().lower()
                    question = f"What is the {location_type_cleaned} office of {org_name}?"
                    questions[question] = {
                        'question': question,
                        'answer': location, 
                        'num_hops': 2,  
                        'num_set_operations': 0,
                        'multiple_answer_dimension': 0,
                        'document_number': doc_num
                    }

        return list(questions.values())

    def generate_all_questions(self):
        """Generate questions from all TTL files"""
        all_questions = []
        
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) 
                    if not f.endswith("ontology.ttl")]
        
        for ttl_file in ttl_files:
            try:
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                graph = self.load_graph(ttl_file)
                all_questions.extend(self.generate_person_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_organization_representative_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_org_role_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_company_by_role_questions_plural(graph, doc_num))
                all_questions.extend(self. generate_org_location_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_location_type_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_person_organization_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_shared_position_questions(graph, doc_num))
                all_questions.extend(self.generate_shared_role_questions(graph, doc_num))
                all_questions.extend(self.generate_shared_role_subrole_questions(graph, doc_num))
                all_questions.extend(self.generate_position_organization_questions(graph, doc_num))
                all_questions.extend(self.generate_person_company_role_question(graph, doc_num))
                all_questions.extend(self.generate_location_role_questions(graph, doc_num))
                all_questions.extend(self.generate_location_office_questions(graph, doc_num))

            except Exception as e:
                print(f"Error processing {ttl_file}: {e}")
        if all_questions:
            df = pd.DataFrame(all_questions)
            
            # Define the desired column order
            columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
            
            # Reorder and ensure all expected columns exist
            df = df.reindex(columns=columns_order)
        else:
            df = pd.DataFrame(columns=['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension'])

        return df

       

def main():
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level2.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe.csv")

if __name__ == "__main__":
    main()
