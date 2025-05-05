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
    
    def get_location_org_people_positions(self, graph):
        """Get all locations, their organizations, and people with positions"""
        location_org_people = {}
        query = """
        SELECT DISTINCT ?loc ?org ?person ?position
        WHERE {
            ?loc a <http://example.org/base/Location> .
            ?org <http://example.org/relation/hasLocationAt> ?loc .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                   <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row[0]
            org_uri = row[1]
            person_uri = row[2]
            position_uri = row[3]
            location = self.clean_uri(loc_uri)
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            if location not in location_org_people:
                location_org_people[location] = {}
            if org_name not in location_org_people[location]:
                location_org_people[location][org_name] = {}
            if person_name not in location_org_people[location][org_name]:
                location_org_people[location][org_name][person_name] = set()
            location_org_people[location][org_name][person_name].add(position)
            
        return location_org_people
    
    def _get_location_types(self, graph):
        """Helper to fetch location-type mappings"""
        loc_types = defaultdict(list)
        
        query = """
        SELECT DISTINCT ?loc (GROUP_CONCAT(?type; separator="|") as ?types)
        WHERE {
            ?loc a <http://example.org/base/Location> ;
                <http://example.org/isInstanceOf/> ?type .
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?loc
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row[0]
            type_uris = row[1].split('|') if row[1] else []
            loc_name = self.clean_uri(loc_uri)
            types = [self.clean_uri(t) for t in type_uris]
            loc_types[loc_name] = types
        
        return loc_types
    
    def get_org_employees_positions(self, graph):
        """Get all organizations with their employees and all positions in the org"""
        org_info = {}
        
        # SPARQL query to get organizations, all their employees and all positions in the org
        query = """
        SELECT DISTINCT ?org ?employee ?position ?position_holder
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?employee <http://example.org/relation/isEmployedBy> ?org .
            ?position_holder <http://example.org/relation/isEmployedBy> ?org ;
                           <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            employee_uri = row[1]
            position_uri = row[2]
            position_holder_uri = row[3]
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri)
            position = self.clean_uri(position_uri)
            position_holder = self.clean_uri(position_holder_uri)
            
            # Initialize organization entry if not exists
            if org_name not in org_info:
                org_info[org_name] = {
                    'employees': set(),
                    'positions': {},  # position -> set of holders
                }
            
            # Add employee
            org_info[org_name]['employees'].add(employee_name)
            
            # Add position and its holder
            if position not in org_info[org_name]['positions']:
                org_info[org_name]['positions'][position] = set()
            org_info[org_name]['positions'][position].add(position_holder)
            
        return org_info
    
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
    
    def _get_org_roles_with_subroles(self, graph):
        """Get all organizations and their roles, including subroles."""
        org_roles = {}

        query = """
            SELECT DISTINCT ?org ?roleOrSubrole ?parentRole
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/isInstanceOf/> ?roleOrSubrole .

                # Check for subroles by looking at the rdfs:subClassOf relationship
                OPTIONAL {
                    ?roleOrSubrole rdfs:subClassOf ?parentRole .
                    FILTER(STRSTARTS(STR(?parentRole), "http://example.org/org_role/"))
                }

                # Bind the final role, prioritizing parent roles over subroles
                BIND(COALESCE(?parentRole, ?roleOrSubrole) AS ?finalRole)

                # Filter to include only roles that start with org_role or org_sub_role
                FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/") ||
                    STRSTARTS(STR(?finalRole), "http://example.org/org_sub_role/"))
            }
        """

        results = graph.query(query)

        print("\n=== DEBUG: Extracting Roles and Subroles ===")
        for row in results:
            print(f"Raw Query Row: {row}")  # Prints entire row from query

            org_uri, role_or_subrole_uri, parent_role_uri = row
            org_name = self.clean_uri(org_uri)
            role_or_subrole = self.clean_uri(role_or_subrole_uri)
            parent_role = self.clean_uri(parent_role_uri) if parent_role_uri else None

            print(f"Processing Organization: {org_name}")
            print(f"  Extracted Role or Subrole: {role_or_subrole}")
            print(f"  Extracted Parent Role: {parent_role if parent_role else 'None'}")

            if org_name not in org_roles:
                org_roles[org_name] = set()

            # If there's a parent role, it means this is a subrole
            if parent_role:
                print(f"  -> Assigning Subrole: {role_or_subrole} under Parent Role: {parent_role}")
                org_roles[org_name].add((parent_role, role_or_subrole))
            else:
                print(f"  -> Assigning as Main Role: {role_or_subrole}")
                org_roles[org_name].add((role_or_subrole, None))

        print("\n=== DEBUG: Final Extracted Organization Roles with Subroles ===")
        for org, roles in org_roles.items():
            print(f"{org}: {roles}")

        return org_roles

    
    def get_org_roles_with_subroles(self, graph):
        """Get organizations with normalized roles in 'subrole role' format"""
        org_roles = defaultdict(lambda: defaultdict(set))

        # Query to fetch roles and their parent roles
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?org ?role (GROUP_CONCAT(?parentRole; separator="|") as ?parents)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?role .
            
            OPTIONAL {
                ?role rdfs:subClassOf ?parentRole .
                FILTER(STRSTARTS(STR(?parentRole), "http://example.org/org_role/"))
            }
        }
        GROUP BY ?org ?role
        """

        # Convert results to a list to iterate multiple times
        results = list(graph.query(query))

        # Track parent roles covered by subroles for each org
        covered_parents = defaultdict(set)

        # First pass: Process subroles and combine with parents
        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            parents = row[2].split('|') if row[2] else []

            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            parent_roles = [self.clean_uri(p) for p in parents if p]

            # Handle subroles (e.g., org_sub_role:lead)
            if str(role_uri).startswith("http://example.org/org_sub_role/"):
                for parent in parent_roles:
                    # Format subrole and parent (e.g., "Lead Arranger")
                    subrole = role.replace('_', ' ').title()
                    parent_role = parent.replace('_', ' ').title()
                    normalized = f"{subrole} {parent_role}"
                    org_roles[org_name]['roles'].add(normalized)
                    # Mark parent as covered to avoid duplicates
                    covered_parents[org_name].add(parent_role)

        # Second pass: Add main roles not covered by subroles
        for row in results:
            org_uri = row[0]
            role_uri = row[1]

            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)

            # Skip subroles (already processed)
            if str(role_uri).startswith("http://example.org/org_sub_role/"):
                continue

            # Process main roles (org_role/)
            main_role = role.replace('_', ' ').title()
            if main_role not in covered_parents[org_name]:
                org_roles[org_name]['roles'].add(main_role)

        return org_roles
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations, their employees, and their roles (formatted as 'subrole role' where applicable)."""
        org_info = {}

        # SPARQL query to fetch organizations, employees, and roles with subrole resolution
        query = """
            SELECT DISTINCT ?org ?employee ?orgRole ?orgSubRole
            WHERE {
                # Fetch organizations and their roles
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/isInstanceOf/> ?orgRole .

                # Fetch employees
                OPTIONAL {
                    ?org <http://example.org/relation/hasEmployee> ?employee .
                }

                # Check if the organization role has a subrole
                OPTIONAL {
                    ?orgRole rdfs:subClassOf ?orgSubRole .
                    FILTER(STRSTARTS(STR(?orgSubRole), "http://example.org/org_role/"))
                }

                # Ensure only valid roles are selected
                FILTER(STRSTARTS(STR(?orgRole), "http://example.org/org_role/") ||
                    STRSTARTS(STR(?orgSubRole), "http://example.org/org_role/"))
            }
        """

        results = graph.query(query)
        for row in results:
            org_uri = row[0]
            employee_uri = row[1] if row[1] else None  # Employee is optional
            org_role_uri = row[2]
            org_sub_role_uri = row[3] if row[3] else None  # Subrole is optional

            # Convert URIs into readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri) if employee_uri else None
            org_role = self.clean_uri(org_role_uri)
            org_sub_role = self.clean_uri(org_sub_role_uri) if org_sub_role_uri else None

            # Correctly format roles by merging subrole and role into a single string
            final_role = f"{org_role} {org_sub_role}" if org_sub_role else org_role

            # Initialize org entry if not present
            if org_name not in org_info:
                org_info[org_name] = {'employees': set(), 'roles': set()}

            # Add formatted role to organization
            org_info[org_name]['roles'].add(final_role.strip())  # Ensure no extra spaces

            # Add employee if exists
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
            location = self.decode_location(raw_location)
            if subrole:
                merged_role = f"{subrole} {role}"
                location_map[location][org].add(merged_role)
            else:
                location_map[location][org].add(role)
        
        return location_map

    def decode_location(self, uri: str) -> str:
        """Convert URI-encoded location to human-readable address"""
        try:
            address_part = uri.split("/location/")[-1]
            return (
                unquote(address_part)
                .replace('_', ' ')
                .replace('%20', ' ')
                .replace('%2C', ',')
                .replace('%5Cn', '\n')
                .replace('%C2%A0', ' ')
                .strip()
            )
        except Exception as e:
            print(f"Error decoding location {uri}: {str(e)}")
            return "Unknown"
        
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
    
    def _get_org_locations(self, graph):
        """Helper to fetch organization locations with proper SPARQL syntax"""
        org_locations = defaultdict(list)
        
        # Corrected SPARQL query with proper syntax
        query = """
        SELECT ?org (GROUP_CONCAT(?loc; separator="|") AS ?locs)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?org
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            loc_uris = row[1].split('|') if row[1] else []
            org_name = self.clean_uri(org_uri)
            locations = [self.clean_uri(loc) for loc in loc_uris]
            org_locations[org_name] = locations
        
        return org_locations

    
    def get_org_employees(self, graph):
        """Fetch all organizations and their employees."""
        org_employees = {}
        query = """
            SELECT DISTINCT ?orgName ?employee
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/relation/hasEmployee> ?employee .

                # Extract organization name from its URI
                BIND(REPLACE(STR(?org), "http://example.org/org_name/", "") AS ?orgName)

                # Extract employee name from URI
                BIND(REPLACE(STR(?employee), "http://example.org/person_name/", "") AS ?employee)
            }
        """

        results = graph.query(query)

        # Process the query results
        for row in results:
            print("Processing row:", row)
            org_uri = row[0]
            employee_uri = row[1]

            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri)

            # Initialize organization entry if not present
            if org_name not in org_employees:
                org_employees[org_name] = set()

            # Add employee to the organization
            org_employees[org_name].add(employee_name)

        return org_employees
    
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
        role_subrole_map = self.get_org_roles_with_subroles(graph)
    
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
    
    def generate_company_by_role_questions_plural(self, graph, doc_num):
        """Generate plural questions (multiple companies per role)"""
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
        """What types of location is [Location] (e.g., Headquarters, Trade Operations, etc.)?"""
        questions = []
        loc_types = self._get_location_types(graph)
        
        for loc, types in loc_types.items():
            if len(types) > 1:
                questions.append({
                    'question': f"What types of location is {loc}?",
                    'answer': ", ".join(sorted(types)),
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
        """What company is both the [Org Role (+ Sub-Role) 1] and [Org Role (+ Sub-Role) 2] in the agreement? [if one]"""
        questions = []
        org_roles = self._get_org_roles_with_subroles(graph)
        role_to_subroles = defaultdict(set)
        for org, roles in org_roles.items():
            for role, subrole in roles:
                if subrole is not None:
                    role_to_subroles[role].add(subrole)

        for org, roles in org_roles.items():
            valid_pairs = []
            for role, subrole in roles:
                if (subrole is not None 
                    and role in role_to_subroles 
                    and subrole in role_to_subroles[role]):
                    valid_pairs.append((role, subrole))
            for (role1, subrole1), (role2, subrole2) in itertools.combinations(valid_pairs, 2):
                unique_components = {role1, subrole1, role2, subrole2}
                if len(unique_components) == 4:
                    question = (
                        f"What company is both the {subrole1} {role1} and the {subrole2} {role2} in the agreement?"
                    )
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
        org_info = self.get_org_employees_positions(graph)
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
