import pandas as pd
from rdflib import Graph, Namespace, URIRef
from urllib.parse import unquote
import glob
import os
import itertools
from itertools import combinations
from collections import defaultdict
from typing import Dict, Set

class QuestionGenerator:
    def __init__(self, ttl_dir="extracted_content"):
        self.ttl_dir = ttl_dir
        
        # Define namespaces
        self.person_name = Namespace("http://example.org/person_name/")
        self.person_position = Namespace("http://example.org/person_position/")
        self.isInstanceOf = URIRef("http://example.org/isInstanceOf/")
        
    def load_graph(self, ttl_file):
        """Load a TTL file into an RDFlib Graph"""
        g = Graph()
        g.parse(ttl_file, format="turtle")
        return g
    
    def clean_uri(self, uri):
        """Clean URI to get readable text"""
        # Remove namespace
        text = str(uri).split('/')[-1]
        # URL decode
        text = text.replace('%20', ' ').replace('%2C', ',').replace('%5Cn', '')
        text = text.replace('_', ' ').replace('%C2%A0', ' ')
        return text.strip()
    
    def get_person_positions(self, graph):
        """Get all persons and their positions from the graph"""
        person_positions = {}
        
        # SPARQL query to get all persons and their positions
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
            
            # Clean the URIs to get readable text
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris)
            
            person_positions[person_name] = positions
            
        return person_positions
    
    def get_org_people_positions(self, graph):
        """Get all organizations and their people with positions"""
        org_people_positions = {}
        
        # SPARQL query to get organizations, their employees and their positions
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
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris)
            
            # Initialize organization entry if not exists
            if org_name not in org_people_positions:
                org_people_positions[org_name] = {}
            
            # Store person's positions
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
    
    def get_location_org_people_positions(self, graph):
        """Get all locations, their organizations, and people with positions"""
        location_org_people = {}
        
        # SPARQL query to get locations, organizations, people and their positions
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
            
            # Clean the URIs to get readable text
            location = self.clean_uri(loc_uri)
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Initialize location entry if not exists
            if location not in location_org_people:
                location_org_people[location] = {}
            
            # Initialize organization entry if not exists
            if org_name not in location_org_people[location]:
                location_org_people[location][org_name] = {}
            
            # Initialize person entry if not exists
            if person_name not in location_org_people[location][org_name]:
                location_org_people[location][org_name][person_name] = set()
            
            # Add position
            location_org_people[location][org_name][person_name].add(position)
            
        return location_org_people

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
            org_info[org_name]['employees'].add(employee_name)
            if position not in org_info[org_name]['positions']:
                org_info[org_name]['positions'][position] = set()
            org_info[org_name]['positions'][position].add(position_holder)
            
        return org_info
    
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
    
    def get_role_org_mapping(self, graph):
            """Get mapping of roles/subroles to their organizations"""
            role_org_map = {}
            
            # SPARQL query to get organizations and their roles/subroles
            query = """
            SELECT DISTINCT ?org ?role ?sub_role
            WHERE {
                ?org a <http://example.org/base/Organization> .
                {
                    ?org <http://example.org/isInstanceOf/> ?role .
                    FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
                }
                OPTIONAL {
                    ?org <http://example.org/isInstanceOf/> ?sub_role .
                    ?sub_role rdfs:subClassOf ?role .
                    FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
                }
            }
            """
            
            results = graph.query(query)
            
            for row in results:
                org_uri = row[0]
                role_uri = row[1]
                sub_role_uri = row[2] if row[2] else None
                
                # Clean the URIs to get readable text
                org_name = self.clean_uri(org_uri)
                role = self.clean_uri(role_uri)
                sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
                
                # Create key for role combination
                role_key = (role, sub_role) if sub_role else (role, None)
                
                # Store organization for this role combination
                if role_key not in role_org_map:
                    role_org_map[role_key] = set()
                role_org_map[role_key].add(org_name)
                    
            return role_org_map
    
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
    

    def generate_shared_position_questions_plural(self, graph, doc_num):
        """[LEVEL 3] What are the positions held by both [Person Name 1] and [Person Name 2]?"""
        questions = []
        person_positions = self.get_person_positions(graph)
        
        # Generate all unique pairs of people
        for person1, person2 in combinations(person_positions.keys(), 2):
            positions1 = person_positions[person1]
            positions2 = person_positions[person2]
            shared_positions = positions1 & positions2 
            
            # Check for AT LEAST 2 shared positions
            if len(shared_positions) >= 2:  
                answer = ", ".join(sorted(shared_positions))
                question = f"What are the positions held by both {person1} and {person2}?"
                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 1, 
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })
        
        return questions
    
    def generate_shared_role_questions_plural(self, graph, doc_num):
        """[Level 3] What roles do both [Organization 1] and [Organization 2] have in the agreement?"""
        questions = []
        org_roles = self.get_org_roles(graph)
        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]
            if len(roles1) > 1 and len(roles2) > 1:
                shared_roles = roles1 & roles2
                if len(shared_roles) > 2:
                    answer = ", ".join(sorted(shared_roles))
                    question = f"What roles do both {org1} and {org2} have in the agreement?"              
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,  
                        'num_set_operations': 1,  
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1
                    })
        
        return questions
    
    def generate_shared_role_subrole_questions_plural(self, graph, doc_num):
        """[Level 3] What companies are both the [Org Role (+ Sub-Role) 1] and [Org Role (+ Sub-Role) 2] in the agreement?"""
        questions = []
        org_roles = self._get_org_roles_with_subroles(graph)
    
        role_to_subroles = defaultdict(set)
        for org, roles in org_roles.items():
            for role, subrole in roles:
                if subrole and subrole != role:  
                    role_to_subroles[role].add(subrole)
        combination_to_orgs = defaultdict(set)
        
        for org, roles in org_roles.items():
            valid_pairs = []
            for role, subrole in roles:
                if (subrole 
                    and role in role_to_subroles 
                    and subrole in role_to_subroles[role]):
                    valid_pairs.append((role, subrole))
            
            seen_combinations = set()
            for (role1, subrole1), (role2, subrole2) in itertools.combinations(valid_pairs, 2):
                if len({role1, subrole1, role2, subrole2}) == 4:
                    combo_key = tuple(sorted([(role1, subrole1), (role2, subrole2)]))
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        combination_to_orgs[combo_key].add(org)
        
        for combo, orgs in combination_to_orgs.items():
            if len(orgs) > 1:
                (role1, subrole1), (role2, subrole2) = combo
                question = (
                    f"Which companies are both the {subrole1} {role1} and the {subrole2} {role2} in the agreement?"
                )
                questions.append({
                    'question': question,
                    'answer': ", ".join(sorted(orgs)),
                    'num_hops': 1,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1  
                })
        
        return questions

    def generate_position_comparison_questions(self, graph, doc_num):
        """What is the position held by [Person Name 1] but not by [Person Name 2]? [if one]"""
        questions = []
        person_positions = self.get_person_positions(graph)

        for person1, person2 in combinations(person_positions.keys(), 2):
            pos1 = person_positions[person1]
            pos2 = person_positions[person2]
            common = pos1 & pos2

            # Only if both have >1 positions, share something, and exactly one unique
            unique = pos1 - pos2
            if len(pos1) > 1 and len(pos2) > 1 and common and len(unique) == 1:
                only_pos = next(iter(unique))
                q_text = f"What is the position held by {person1} but not by {person2}?"
                questions.append({
                    'question': q_text,
                    'answer': only_pos,
                    'num_hops': 1,
                    'num_set_operations': 2,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions
    
    
    
    def generate_org_role_comparison_questions(self, graph, doc_num):
        """
        What role does [Org Name 1] have in the agreement which is not the role of [Org Name 2]?
        """
        questions = []
        org_roles = self.get_org_roles_and_subroles(graph)

        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]

            # Must both have multiple roles and share at least one
            common = set(roles1.keys()) & set(roles2.keys())
            if not common or len(roles1) <= 1 or len(roles2) <= 1:
                continue

            # Build unique role/subrole combos for org1
            unique_combos = []

            # Roles that org2 doesn’t have at all
            for role in set(roles1) - set(roles2):
                subs = roles1[role]
                if subs:
                    unique_combos.extend(f"{sub} {role}" for sub in subs)
                else:
                    unique_combos.append(role)

            # Shared roles but with extra subroles in org1
            for role in common:
                extra_subs = roles1[role] - roles2[role]
                for sub in extra_subs:
                    unique_combos.append(f"{sub} {role}")

            if len(unique_combos) == 1:
                combo = unique_combos[0]
                q = f"What role does {org1} have in the agreement which is not the role of {org2}?"
                questions.append({
                    'question': q,
                    'answer': combo,
                    'num_hops': 2,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions


    def generate_position_organization_questions_plural(self, graph, doc_num):
        """[Level 3] Who are the [Position]s of [Organization]?"""
        questions = {}
        org_info = self.get_org_employees_positions(graph)
        for org_name, info in org_info.items():
            for position, holders in info['positions'].items():
                if len(holders) > 1:  
                    question = f"Who are the {position}s of {org_name}?"
                    answer = ", ".join(sorted(holders))  

                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1  
                    }

        return list(questions.values())
    
    def generate_location_position_questions(self, graph, doc_num):
        """
        Who is the [Person Position] of the company associated with [Location]? [if one]
        """
        questions = []
        # { location: { org: { person: [positions] } } }
        loc_org_people = self.get_location_org_people_positions(graph)

        for location, orgs in loc_org_people.items():
            # Collect all holders per position at this location
            position_holders = defaultdict(set)
            for people in orgs.values():
                for person, positions in people.items():
                    for pos in positions:
                        position_holders[pos].add(person)

            # Only singleton positions
            for pos, holders in position_holders.items():
                if len(holders) == 1:
                    person = next(iter(holders))
                    q = f"Who is the {pos} of the company associated with {location}?"
                    questions.append({
                        'question': q,
                        'answer': person,
                        'num_hops': 3,               # location → org → person → position
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })

        return questions


        

    def generate_position_organization_questions_pair(self, graph, doc_num):
            """Who is the [Person Position 1] and [Person Position 2] of [Org Name]? [if one]"""
            questions = []
            org_info = self.get_org_employees_positions(graph)
            
            for org_name, info in org_info.items():
                clean_org = self.clean_uri(org_name)
                holder_positions = defaultdict(set)

                # Phase 1: Collect positions with single holders
                for raw_pos, holders in info['positions'].items():
                    if len(holders) == 1:
                        holder = self.clean_uri(next(iter(holders)))
                        clean_pos = self.clean_uri(raw_pos).replace(',,', ',').rstrip(',').title()
                        
                        if holder and clean_pos:
                            holder_positions[holder].add(clean_pos)

                seen_pairs = set()
                for holder, positions in holder_positions.items():
                    if len(positions) >= 2:
                        # Sort positions for consistent ordering
                        sorted_positions = sorted(positions)
                        
                        # Generate all unique pairs
                        for pos1, pos2 in combinations(sorted_positions, 2):
                            pair_key = f"{pos1}|{pos2}"
                            
                            # Prevent duplicate questions
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                
                                # Build question and answer
                                question = f"Who is the {pos1} and {pos2} of {clean_org}?"
                                questions.append({
                                    'question': question,
                                    'answer': holder,
                                    'num_hops': 2,
                                    'num_set_operations': 1,
                                    'document_number': doc_num,
                                    'multiple_answer_dimension': 0
                                })
            
            return questions
    def generate_employee_org_position_questions(self, graph, doc_num):
        """
        Who is the [Person Position] of the company where [Person Name] is employed?
        """
        questions = []
        org_info = self.get_org_employees_positions(graph)
        
        for org_name, info in org_info.items():
            for employee in info['employees']:
                for position, holders in info['positions'].items():
                    # skip positions held by the employee themself
                    if employee in holders:
                        continue
                    # only singleton-holder positions
                    if len(holders) == 1:
                        person = next(iter(holders))
                        q = f"Who is the {position} of the company where {employee} is employed?"
                        questions.append({
                            'question': q,
                            'answer': person,
                            'num_hops': 3,               # person → org → position
                            'num_set_operations': 0,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        })
        return questions


    def get_org_role_subrole_map(self, graph):
        """Get mapping of org_roles to their sub_roles"""
        role_subrole_map = {}
        
        # SPARQL query to get role-subrole relationships
        query = """
        SELECT DISTINCT ?role ?sub_role
        WHERE {
            ?sub_role rdfs:subClassOf ?role .
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            role_uri = row[0]
            sub_role_uri = row[1]
            
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri)
            
            if role not in role_subrole_map:
                role_subrole_map[role] = set()
            role_subrole_map[role].add(sub_role)
            
        return role_subrole_map
    
    def get_org_roles_and_subroles(self, graph):
        """Get all organizations with their roles and subroles"""
        org_roles = {}
        
        # SPARQL query to get organizations and their roles/subroles
        query = """
        SELECT DISTINCT ?org ?role ?sub_role
        WHERE {
            ?org a <http://example.org/base/Organization> .
            {
                ?org <http://example.org/isInstanceOf/> ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }
            OPTIONAL {
                ?org <http://example.org/isInstanceOf/> ?sub_role .
                ?sub_role rdfs:subClassOf ?role .
                FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
            }
        }
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            sub_role_uri = row[2] if row[2] else None
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            
            # Initialize organization entry if not exists
            if org_name not in org_roles:
                org_roles[org_name] = {}
            
            # Store role and its subrole if exists
            if role not in org_roles[org_name]:
                org_roles[org_name][role] = set()
            if sub_role:
                org_roles[org_name][role].add(sub_role)
                
        return org_roles

 
    
    def generate_person_company_role_question_plural(self, graph, doc_num):
        """What are the roles in the agreement of the company where [Person Name] is employed?"""
        questions = {}
        org_info = self.get_org_employees_and_roles(graph)
        print(org_info)

        for org_name, info in org_info.items():
            roles = info['roles']  
            if len(roles) > 1:  
                sorted_roles = ", ".join(sorted(roles))  
                for employee in info['employees']:
                    question = f"What are the roles in the agreement of the company where {employee} is employed?"
                    answer = sorted_roles  
                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1  
                    }

        return list(questions.values())
    
    def generate_location_role_questions_plural(self, graph, doc_num):
        """[Level 3] What are the roles in the agreement of the company associated with [Location]?"""
        questions = {}
        location_data = self.get_location_org_roles(graph)
        
        for location, orgs in location_data.items():
            all_roles = set()
            for roles in orgs.values():
                all_roles.update(roles)

            if len(all_roles) > 1:
                sorted_roles = ", ".join(sorted(all_roles))  
                question = f"What are the roles in the agreement of the company associated with {location}?"
                
                questions[question] = {
                    'question': question,
                    'answer': sorted_roles,
                    'num_hops': 2,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1 
                }
        
        return list(questions.values())
    

    def generate_org_role_subrole_comparison_questions(self, graph, doc_num):
        """
        What company is the [Org Role (+ Sub-Role) 1] but not the [Org Role (+ Sub-Role) 2] in the agreement? [if one]
        """
        questions = []
        role_org_map = self.get_role_org_mapping(graph)

        # Iterate over all pairs of (role, subrole) combos
        for role_pair1, role_pair2 in combinations(role_org_map.keys(), 2):
            orgs1 = role_org_map[role_pair1]
            orgs2 = role_org_map[role_pair2]

            # Need both combos to have multiple orgs and share at least one
            if len(orgs1) > 1 and len(orgs2) > 1 and orgs1 & orgs2:
                unique_orgs = orgs1 - orgs2

                # Exactly one unique org → singular question
                if len(unique_orgs) == 1:
                    role1, sub1 = role_pair1
                    role2, sub2 = role_pair2

                    # Build “Subrole Role” strings
                    role1_str = f"{sub1} {role1}" if sub1 else role1
                    role2_str = f"{sub2} {role2}" if sub2 else role2

                    answer = next(iter(unique_orgs))
                    question = (
                        f"What company is the {role1_str} but not the {role2_str} in the agreement?"
                    )

                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 1,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })

        return questions

    
    def generate_person_position_by_org_role_questions(self, graph, doc_num):
        """
        Who is the [Position] of the company which is the [Org Role (+ Sub‑Role)] in the agreement?
        """
        questions = []
        # 1) Build org→normalized-roles map
        org_roles = self.get_org_roles_with_subroles(graph)  # { org_name: {'roles': set([...])} }
        # 2) Invert to role→{orgs}
        role_to_org = defaultdict(set)
        for org, data in org_roles.items():
            for role in data['roles']:
                role_to_org[role].add(org)

        # 3) Fetch position holders for each org once
        org_positions = self.get_org_employees_positions(graph)  
        #    → returns { org_name: {'positions': {pos: {holder1, …}}}, … }

        for role, orgs in role_to_org.items():
            # only consider roles that map to exactly one org
            if len(orgs) != 1:
                continue
            org = next(iter(orgs))
            info = org_positions.get(org, {})
            positions = info.get('positions', {})  
            
            # pick exactly those positions held by exactly one person
            singleton = {pos: next(iter(holders))
                        for pos, holders in positions.items()
                        if len(holders) == 1}
            
            # only if there is exactly one such position
            if len(singleton) == 1:
                pos, person = next(iter(singleton.items()))
                q = f"Who is the {pos} of the company which is the {role} in the agreement?"
                questions.append({
                    'question': q,
                    'answer': person,
                    'num_hops': 2,             
                    'num_set_operations': 1,    
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })
            

        return questions
    

    def generate_location_type_by_org_role_questions(self, graph, doc_num):
        """
        What is the address of [Location Type] of the company which is the [Org Role (+ Sub‑Role)] in the agreement?)
        """
        questions = []

        # 1) Build role(+subrole) → {org_name} map
        org_roles = self.get_org_roles_with_subroles(graph)  # { org_name: {'roles': set([...])} }
        role_to_org = defaultdict(set)
        for org, data in org_roles.items():
            for role in data['roles']:
                role_to_org[role].add(org)

        # 2) Fetch org_name → single location‑type
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?type; separator="|") AS ?types)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/relation/hasLocationAt> ?loc .
            ?loc <http://example.org/isInstanceOf/> ?type .
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?org
        HAVING (COUNT(?type) = 1)
        """
        loc_results = graph.query(query)

        org_to_loc_type = {}
        for org_uri, types_concat in loc_results:
            org_name = self.clean_uri(org_uri)
            # HAVING=1 guarantees only one type, no split needed
            loc_type_uri = types_concat
            loc_type = self.clean_uri(loc_type_uri)
            org_to_loc_type[org_name] = loc_type

        # 3) For each role that maps to exactly one org, and that org has exactly one loc_type → question!
        for role, orgs in role_to_org.items():
            if len(orgs) != 1:
                continue
            org = next(iter(orgs))
            if org not in org_to_loc_type:
                continue

            loc_type = org_to_loc_type[org]
            q_text = (
                f"What is the address of {loc_type} of the company which is the {role} in the agreement?"
            )
            questions.append({
                'question': q_text,
                'answer': org,  
                'num_hops': 3,               
                'num_set_operations': 0,     
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })

        return questions
    
    def generate_address_of_location_type_by_person_questions(self, graph, doc_num):
        """
        What is the address of [LocationType] of the company where [PersonName] is employed?
        """
        questions = []

        q_person_org = """
        SELECT ?person (SAMPLE(?org) AS ?org)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                    <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?person
        HAVING (COUNT(?org) = 1)
        """
        person_to_org = {
            self.clean_uri(p): self.clean_uri(o)
            for p, o in graph.query(q_person_org)
        }

        # 2) Build org → { loc_type → {locations} } map
        loc_data = self.get_location_org_data(graph)
        org_to_loctypes = defaultdict(lambda: defaultdict(set))
        for location, orgs in loc_data.items():
            for org_name, info in orgs.items():
                for loc_type in info['location_types']:
                    org_to_loctypes[org_name][loc_type].add(location)

        # 3) Generate questions for each person/org
        for person, org_name in person_to_org.items():
            loc_types = org_to_loctypes.get(org_name, {})
            for loc_type, locations in loc_types.items():
                # only when exactly one location exists for that type
                if len(locations) == 1:
                    address = next(iter(locations))
                    q_text = (
                        f"What is the address of {loc_type} "
                        f"of the company where {person} is employed?"
                    )
                    questions.append({
                        'question': q_text,
                        'answer': address,
                        'num_hops': 3,               
                        'num_set_operations': 0,     
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })

        return questions

    

    def generate_all_questions(self):
        """Generate questions from all TTL files"""
        all_questions = []
        
        # Get all TTL files except ontology.ttl
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) 
                    if not f.endswith("ontology.ttl")]
        
        for ttl_file in ttl_files:
            try:
                # Extract document number from filename
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                
                # Load the graph
                graph = self.load_graph(ttl_file)
                all_questions.extend(self.generate_shared_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_shared_role_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_shared_role_subrole_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_position_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_org_role_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_org_role_subrole_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_position_organization_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_person_company_role_question_plural(graph, doc_num))
                all_questions.extend(self.generate_location_role_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_position_organization_questions_pair(graph, doc_num))
                all_questions.extend(self.generate_person_position_by_org_role_questions(graph, doc_num))
                all_questions.extend(self.generate_location_position_questions(graph, doc_num))
                all_questions.extend(self.generate_employee_org_position_questions(graph, doc_num))
                all_questions.extend(self.generate_location_type_by_org_role_questions(graph, doc_num))
                all_questions.extend(self.generate_address_of_location_type_by_person_questions(graph, doc_num))
           
            except Exception as e:
                print(f"Error processing {ttl_file}: {str(e)}")
        
        df = pd.DataFrame(all_questions)
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
        df = df[columns_order]
        
        return df

def main():
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level3.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe_L3.csv")

if __name__ == "__main__":
    main()
