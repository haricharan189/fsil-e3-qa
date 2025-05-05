import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from itertools import combinations
from collections import defaultdict


class QuestionGenerator:
    def __init__(self, ttl_dir="extracted_content"):
        self.ttl_dir = ttl_dir
        
        # Define namespaces
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
    
    def get_org_roles_with_subroles(self, graph):
        """
        Get all organizations and their (base_role, subrole) pairs,
        but only emit a (role, None) if it truly came from org_role/.
        """
        org_roles = defaultdict(set)

        query = """
            SELECT DISTINCT ?org ?roleOrSubrole ?parentRole
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/isInstanceOf/> ?roleOrSubrole .

                OPTIONAL {
                    ?roleOrSubrole rdfs:subClassOf ?parentRole .
                    FILTER(STRSTARTS(STR(?parentRole), "http://example.org/org_role/"))
                }

                BIND(COALESCE(?parentRole, ?roleOrSubrole) AS ?finalRole)

                FILTER(
                    STRSTARTS(STR(?finalRole), "http://example.org/org_role/") ||
                    STRSTARTS(STR(?finalRole), "http://example.org/org_sub_role/")
                )
            }
        """
        results = graph.query(query)

        for org_uri, role_uri, parent_uri in results:
            org_name  = self.clean_uri(org_uri)
            role_name = self.clean_uri(role_uri)
            # only clean the parent if present
            parent_name = self.clean_uri(parent_uri) if parent_uri else None

            if parent_name:
                # we saw a subClassOf parent, so this is a true subrole
                org_roles[org_name].add((parent_name, role_name))
            else:
                # only add a base-role entry if it really came from org_role/, not org_sub_role/
                if str(role_uri).startswith("http://example.org/org_role/"):
                    org_roles[org_name].add((role_name, None))

        return dict(org_roles)

    
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations, their employees, and their roles in the organization."""
        org_info = {}

        # SPARQL query to fetch organizations and their associated employees (isEmployedBy)
        query = """
                SELECT DISTINCT ?orgName ?employee (GROUP_CONCAT(DISTINCT ?finalRoleLabel; separator="|") as ?roles)
            WHERE {
        # Fetch organization names
        ?org a <http://example.org/base/Organization> ;
            <http://example.org/isInstanceOf/> ?roleOrSubrole ;
            <http://example.org/relation/hasEmployee> ?employee .

        # Extract organization name from its URI
        BIND(REPLACE(STR(?org), "http://example.org/org_name/", "") AS ?orgName)

        # Extract employee name from URI (since no rdfs:label exists)
        BIND(REPLACE(STR(?employee), "http://example.org/person_name/", "") AS ?employee)

        # Handle sub-role and direct role cases
        OPTIONAL {
            ?roleOrSubrole rdfs:subClassOf ?role .
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
        }

        # Resolve the final role
        BIND(COALESCE(?role, ?roleOrSubrole) AS ?finalRole)

        # Ensure only valid roles are selected
        FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/"))

        # Get role label, fallback to extracting name from URI if not found
        OPTIONAL { ?finalRole rdfs:label ?roleLabel }
        BIND(COALESCE(?roleLabel, REPLACE(STR(?finalRole), "http://example.org/org_role/", "")) AS ?finalRoleLabel)
    }
    GROUP BY ?orgName ?employee

        """
        results = graph.query(query)
        # Process the query results to group employees and roles by organization
        for row in results:
            print("Processing row: ", row)
            org_uri = row[0]
            employee_uri = row[1]
            role_uris = row[2].split('|') if row[2] else []

            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            employee_name = self.clean_uri(employee_uri)
            roles = set(self.clean_uri(role) for role in role_uris)
            if org_name not in org_info:
                org_info[org_name] = {'employees': set(), 'roles': set()}
            
            # Add employee and roles to the organization's entry
            org_info[org_name]['employees'].add(employee_name)
            org_info[org_name]['roles'].update(roles)

        return org_info
    
    def get_org_roles_and_locations(self, graph):
        """Get all organizations, their roles, and locations from the graph."""
        org_data = {}

        query = """
            SELECT DISTINCT ?org ?location (GROUP_CONCAT(DISTINCT ?finalRole; separator="|") as ?roles)
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/isInstanceOf/> ?roleOrSubrole ;
                    <http://example.org/relation/hasLocationAt> ?location .

                OPTIONAL {
                    ?roleOrSubrole rdfs:subClassOf ?role .
                    FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
                }

                BIND(COALESCE(?role, ?roleOrSubrole) AS ?finalRole)
                FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/"))
            }
            GROUP BY ?org ?location
        """

        results = graph.query(query)

        for row in results:
            org_uri = row[0]
            location_uri = row[1]
            role_uris = row[2].split('|')

            # Clean the URIs
            org_name = self.clean_uri(org_uri)
            location = self.clean_uri(location_uri)
            roles = set(self.clean_uri(role) for role in role_uris)

            if location not in org_data:
                org_data[location] = {}

            org_data[location][org_name] = roles

        return org_data
    
    def get_location_org_data(self, graph):
        """Retrieve organizations, their associated locations, and location types."""
        location_org_data = {}

        # SPARQL query to get organizations, locations, and location types
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
    
    def get_org_role_subrole_mapping(self, graph):
        """Get mapping of organizations to role-subrole mappings."""
        
        # Step 1: Get role-subrole relationships
        role_subrole_map = self.get_org_role_subrole_map(graph)  
        
        # Step 2: Initialize organization-role-subrole mapping
        org_role_subrole_mapping = {}

        # SPARQL query to get organizations and their roles
        query = """
        SELECT DISTINCT ?org ?role
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?role .
        }
        """

        results = graph.query(query)

        for row in results:
            org_uri, role_uri = row

            # Clean URIs
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)

            # Initialize org entry if not exists
            if org_name not in org_role_subrole_mapping:
                org_role_subrole_mapping[org_name] = {}

            # Assign role_subrole_map to the organization
            org_role_subrole_mapping[org_name][role] = role_subrole_map.get(role, None)

        return org_role_subrole_mapping
    
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
    
    def generate_position_comparison_questions_plural(self, graph, doc_num):
        """
        What are the positions held by [Person Name 1] but not by [Person Name 2]? 
        """
        questions = []
        person_positions = self.get_person_positions(graph)

        for person1, person2 in combinations(person_positions.keys(), 2):
            pos1 = person_positions[person1]
            pos2 = person_positions[person2]
            common = pos1 & pos2

            # Only if both have >1 positions, share something, and more than one unique
            unique = pos1 - pos2
            if len(pos1) > 1 and len(pos2) > 1 and common and len(unique) > 1:
                answer = ", ".join(sorted(unique))
                q_text = f"What are the positions held by {person1} but not by {person2}?"
                questions.append({
                    'question': q_text,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 2,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions
    
    def generate_org_role_comparison_questions_plural(self, graph, doc_num):
        """
        What roles does [Org Name 1] have in the agreement which are not the roles of [Org Name 2]?
        """
        questions = []
        org_roles = self.get_org_roles_and_subroles(graph)

        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]

            common = set(roles1.keys()) & set(roles2.keys())
            if not common or len(roles1) <= 1 or len(roles2) <= 1:
                continue

            unique_combos = []
            for role in set(roles1) - set(roles2):
                subs = roles1[role]
                if subs:
                    unique_combos.extend(f"{sub} {role}" for sub in subs)
                else:
                    unique_combos.append(role)

            for role in common:
                extra_subs = roles1[role] - roles2[role]
                for sub in extra_subs:
                    unique_combos.append(f"{sub} {role}")

            if len(unique_combos) > 1:
                unique_combos.sort()
                answer = ", ".join(unique_combos)
                q = f"What roles does {org1} have in the agreement which are not the roles of {org2}?"
                questions.append({
                    'question': q,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 2,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions
    
    def generate_org_role_subrole_comparison_questions_plural(self, graph, doc_num):
        """
        What companies are the [Subrole Role] but not the [Subrole Role] in the agreement?
        """
        questions = []
        role_org_map = self.get_role_org_mapping(graph)

        for role_pair1, role_pair2 in combinations(role_org_map.keys(), 2):
            orgs1 = role_org_map[role_pair1]
            orgs2 = role_org_map[role_pair2]

            if len(orgs1) > 1 and len(orgs2) > 1 and orgs1 & orgs2:
                unique_orgs = orgs1 - orgs2

                # More than one unique org → plural question
                if len(unique_orgs) > 1:
                    role1, sub1 = role_pair1
                    role2, sub2 = role_pair2

                    role1_str = f"{sub1} {role1}" if sub1 else role1
                    role2_str = f"{sub2} {role2}" if sub2 else role2

                    answer = ", ".join(sorted(unique_orgs))
                    question = (
                        f"What companies are the {role1_str} but not the {role2_str} in the agreement?"
                    )

                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,
                        'num_set_operations': 2,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1
                    })

        return questions

    def generate_position_comparison_questions(self, graph, doc_num):
        """
        What is the position held by [Person Name 1] but not by [Person Name 2] or [Person Name 3]? [if one]
        """
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)
        print(f"Total people with positions: {len(person_positions)}")

        # Generate all unique combinations of three people
        for person1, person2, person3 in combinations(person_positions.keys(), 3):
            print(f"\nChecking combination: {person1}, {person2}, {person3}")

            # Get positions for each person
            positions1 = person_positions[person1]
            positions2 = person_positions[person2]
            positions3 = person_positions[person3]

            print(f"Positions of {person1}: {positions1}")
            print(f"Positions of {person2}: {positions2}")
            print(f"Positions of {person3}: {positions3}")

            # Ensure person1 and person2 share at least one position
            shared_12 = positions1 & positions2
            if len(shared_12) == 0:
                print(f"Skipping: {person1} and {person2} share no positions.")
                continue

            # Ensure person1 and person3 share at least one position
            shared_13 = positions1 & positions3
            if len(shared_13) == 0:
                print(f"Skipping: {person1} and {person3} share no positions.")
                continue

            print(f"{person1} shares at least one position with {person2}: {shared_12}")
            print(f"{person1} shares at least one position with {person3}: {shared_13}")


            # Find positions unique to person1 (not held by person2 or person3)
            unique_positions = positions1 - (positions2 | positions3)

            print(f"Positions unique to {person1}: {unique_positions}")

            # Only proceed if exactly one unique position exists
            if len(unique_positions) == 1:
                unique_position = next(iter(unique_positions))

                # Create the question
                question = f"What is the position held by {person1} but not by {person2} or {person3}?"
                answer = unique_position

                print(f"Generated question: {question} | Answer: {answer}")

                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1, 
                    'num_set_operations': 3, 
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })
        return questions

    


    def generate_shared_position_exclusion_questions(self, graph, doc_num):
        """
        What is the position held by [Person Name 1] and [Person Name 2] but not by [Person Name 3]? [if one]
        """
        questions = []
        seen_questions = set()  
        person_positions = self.get_person_positions(graph)
        for person1, person2, person3 in combinations(person_positions.keys(), 3):
            print(f"\nChecking combination: {person1}, {person2}, {person3}")
            positions1 = person_positions[person1]
            positions2 = person_positions[person2]
            positions3 = person_positions[person3]

            print(f"Positions of {person1}: {positions1}")
            print(f"Positions of {person2}: {positions2}")
            print(f"Positions of {person3}: {positions3}")
            shared_12 = positions1 & positions2
            if len(shared_12) == 0:
                print(f"Skipping: {person1} and {person2} share no positions.")
                continue

            # Ensure person1 and person3 share at least one position
            shared_13 = positions1 & positions3
            if len(shared_13) == 0:
                print(f"Skipping: {person1} and {person3} share no positions.")
                continue

            print(f"{person1} shares at least one position with {person2}: {shared_12}")
            print(f"{person1} shares at least one position with {person3}: {shared_13}")

            # Find positions shared by person1 and person2 but NOT by person3
            shared_by_two = shared_12 - positions3

            # Only proceed if there is exactly one such shared position
            if len(shared_by_two) == 1:
                shared_position = next(iter(shared_by_two))

                # Sort (person1, person2) to avoid duplicate questions
                key = tuple(sorted([person1, person2]) + [shared_position])
                if key in seen_questions:
                    continue 
                seen_questions.add(key)
                question = f"What is the position held by {person1} and {person2} but not by {person3}?"
                answer = shared_position

                print(f"Generated question: {question} | Answer: {answer}")

                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1,  
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        print("\nFinal generated questions:", questions)
        return questions


    

    def generate_exclusive_shared_role_questions(self, graph, doc_num):
        """
        What role do [Org Name 1] and [Org Name 2] have in the agreement which is not the role of [Org Name 3]? [if one]
        """

        questions = []
        seen_questions = set()  # Set to track unique questions

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)

        # Filter organizations that have more than one role-subrole pair
        valid_orgs = {org: roles for org, roles in org_roles.items() if len(roles) > 1}

        # Generate all unique combinations of three organizations
        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            print(f"\nChecking combination: {org1}, {org2}, {org3}")

            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]

            print(f"Roles of {org1}: {roles1}")
            print(f"Roles of {org2}: {roles2}")
            print(f"Roles of {org3}: {roles3}")

            # Ensure Org1 shares at least one role-subrole pair with Org2
            shared_12 = roles1 & roles2
            if len(shared_12) == 0:
                print(f"Skipping: {org1} and {org2} share no role-subrole pairs.")
                continue

            # Ensure Org1 shares at least one role-subrole pair with Org3
            shared_13 = roles1 & roles3
            if len(shared_13) == 0:
                print(f"Skipping: {org1} and {org3} share no role-subrole pairs.")
                continue

            print(f"{org1} shares at least one role-subrole with {org2}: {shared_12}")
            print(f"{org1} shares at least one role-subrole with {org3}: {shared_13}")

            # Find role-subrole pairs shared by Org1 and Org2 but NOT by Org3
            exclusive_shared_roles = (shared_12 - roles3)

            # Remove pairs where subrole is None
            exclusive_shared_roles = {(role, subrole) for role, subrole in exclusive_shared_roles if subrole is not None}

            if exclusive_shared_roles:
                role_subrole_str = ", ".join([f"{subrole} {role}" for role, subrole in exclusive_shared_roles])

                # Sort (Org1, Org2) to avoid duplicate questions
                key = (frozenset([org1, org2]), frozenset(exclusive_shared_roles), org3)
                if key in seen_questions:
                    continue  # Skip duplicate question
                seen_questions.add(key)

                if len(exclusive_shared_roles) == 1:
                    question = f"What is the role and subrole that {org1} and {org2} share in the agreement that is not held by {org3}?"
                
                    print(f"Generated question: {question} | Answer: {role_subrole_str}")

                    questions.append({
                    'question': question,
                    'answer': role_subrole_str,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1 if len(exclusive_shared_roles) > 1 else 0
                    })

        print("\nFinal generated questions:", questions)
        return questions



    
    
    def generate_exclusive_role_subrole_questions(self, graph, doc_num):
        """
        What role does [Org Name 1] have in the agreement which is not the role of [Org Name 2] or [Org Name 3]? [if one]
        """

        questions = []
        seen_questions = set()

        org_roles = self.get_org_roles_with_subroles(graph)

        # Remove (role, subrole) pairs where subrole is None or "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.strip().lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Process organizations with multiple role-subrole pairs
        orgs_with_multiple_roles = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) > 1}

        for org1, org2, org3 in combinations(orgs_with_multiple_roles.keys(), 3):
            roles1 = orgs_with_multiple_roles[org1]
            roles2 = orgs_with_multiple_roles[org2]
            roles3 = orgs_with_multiple_roles[org3]

            # Ensure Org A shares at least one role-subrole with Org B
            shared_12 = roles1 & roles2
            if not shared_12:
                continue  

            # Ensure Org A shares at least one role-subrole with Org C
            shared_13 = roles1 & roles3
            if not shared_13:
                continue  

            # Find a **single** role-subrole **exclusively** held by Org1 but NOT by Org2 or Org3
            exclusive_roles = roles1 - (roles2 | roles3)

            # Handle cases where subrole is None
            exclusive_roles = {(role, subrole) for role, subrole in exclusive_roles if subrole is not None}

            if len(exclusive_roles) == 1:  # Ensure exactly one exclusive role-subrole pair
                (role, subrole) = list(exclusive_roles)[0]

                # If no subrole exists, just use the role
                answer = f"{subrole} {role}" if subrole else role

                question = (f"What is the role and subrole that {org1} holds in the agreement that is not held by {org2} or {org3}?")

                key = (org1, role, subrole, frozenset([org2, org3]))
                if key in seen_questions:
                    continue  
                seen_questions.add(key)

                print(f"Singular Question: {question}")
                print(f"Answer: {answer}")

                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        print("\nFinal Generated Singular Questions:")
        for q in questions:
            print(f"  - {q['question']} | Answer: {q['answer']}")

        return questions
    
    def generate_shared_role_subrole_exclusive_questions(self, graph, doc_num):
        """What company is the [Role/Subrole1] and [Role/Subrole2] but not [Role/Subrole3]?"""
        questions = []
        org_roles = self.get_org_roles_and_subroles(graph)
        
        # Build (role, subrole) → orgs mapping including base roles
        role_sub_orgs = defaultdict(set)
        for org, role_map in org_roles.items():
            for role, subs in role_map.items():
                # Add base role marker
                role_sub_orgs[(role, None)].add(org)
                # Add subroles
                for sub in subs:
                    role_sub_orgs[(role, sub)].add(org)
        
        # Generate valid triple combinations
        for (r1, s1), (r2, s2), (r3, s3) in combinations(role_sub_orgs.keys(), 3):
            # Get organizations with first two role/subs
            candidates = role_sub_orgs[(r1, s1)] & role_sub_orgs[(r2, s2)]
            # Exclude orgs with third role/sub
            final = candidates - role_sub_orgs[(r3, s3)]
            
            if len(final) == 1:
                # Format role/sub displays
                def format_role_sub(role, sub):
                    return f"{sub} {role}".strip() if sub else role
                    
                q = (
                    f"What company is the {format_role_sub(r1, s1)} "
                    f"and {format_role_sub(r2, s2)} but not the "
                    f"{format_role_sub(r3, s3)} in the agreement?"
                )
                
                questions.append({
                    'question': q,
                    'answer': next(iter(final)),
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })
        
        return questions
    

    def generate_org_role_exclusion_questions(self, graph, doc_num):
        """
        What company is the [Role/Subrole1] but not the [Role/Subrole2] and [Role/Subrole3] in the agreement?
        """

        questions = []
        # 1) org → {(role, subrole)}
        org_roles = self.get_org_roles_with_subroles(graph)

        # 2) invert to (role, subrole) → set(orgs)
        pair_to_orgs = defaultdict(set)
        for org, pairs in org_roles.items():
            for role, sub in pairs:
                pair_to_orgs[(role, sub)].add(org)

        # helper to format “Subrole Role” or just “Role”
        def fmt(role, sub):
            return f"{sub} {role}".strip() if sub else role

        # 3) for each triple (A, B, C):
        for (r1, s1), (r2, s2), (r3, s3) in combinations(pair_to_orgs.keys(), 3):
            include_orgs = pair_to_orgs[(r1, s1)]
            exclude_orgs = pair_to_orgs[(r2, s2)] | pair_to_orgs[(r3, s3)]

            # *New filter*: ensure at least one org with A also has B or C
            if not (include_orgs & exclude_orgs):
                continue

            # those with A but not B/C
            exclusive = include_orgs - exclude_orgs
            if len(exclusive) == 1:
                org = next(iter(exclusive))
                q_text = (
                    f"What company is the {fmt(r1, s1)} "
                    f"but not the {fmt(r2, s2)} or {fmt(r3, s3)} in the agreement?"
                )
                questions.append({
                    'question': q_text,
                    'answer': org,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions


    def generate_person_position_questions_for_unique_dual_role(self, graph, doc_num):
        """Who is the [Person Position] of the company which is both the [Org Role(-s) (+ Sub-Role(-s)) 1] and the [Org Role(-s) (+ Sub-Role(-s)) 2] in the agreement? [if one, and the company should be uniquely identifiable]."""
        questions = {}  

        # Get all organizations with their employees and positions
        org_info = self.get_org_employees_positions(graph)
        print("\nExtracted org employee positions:", org_info)  # Debugging

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)
        print("\nExtracted org roles with subroles:", org_roles)  # Debugging

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in org_roles.items() if roles}
        print("\nOrganizations after filtering empty roles:", cleaned_org_roles)  # Debugging

        # Identify companies that hold exactly two distinct roles/subroles
        dual_role_companies = {}
        for org, roles in cleaned_org_roles.items():
            if len(roles) == 2:  # Ensure exactly two roles
                dual_role_companies[org] = roles

        print("\nIdentified dual-role companies:", dual_role_companies)  # Debugging

        # Keep track of already generated questions to remove duplicates
        seen_questions = set()

        # Process each uniquely identified dual-role company
        for org_name, roles in dual_role_companies.items():
            print(f"\nProcessing company: {org_name} with roles: {roles}")  # Debugging

            if org_name in org_info:  # Ensure the company has known positions
                for position, holders in org_info[org_name]['positions'].items():
                    print(f"\nProcessing position: {position} with holders: {holders}")  # Debugging

                    if len(holders) == 1:  # Ensure the position has exactly one person
                        role_1, subrole_1 = list(roles)[0]
                        role_2, subrole_2 = list(roles)[1]

                        # Format role-subrole pairs properly
                        role_1_desc = f"{subrole_1} {role_1}" if subrole_1 else role_1
                        role_2_desc = f"{subrole_2} {role_2}" if subrole_2 else role_2

                        # Construct question
                        question = (f"Who is the {position} of the company which is both the {role_1_desc} "
                                    f"and the {role_2_desc} in the agreement?")

                        # Skip if the question has already been generated
                        if question in seen_questions:
                            print(f"Skipping duplicate question: {question}")  # Debugging
                            continue
                        
                        # Mark the question as seen
                        seen_questions.add(question)

                        # Answer is the single known position holder
                        answer = list(holders)[0]

                        print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                        # Store the question and answer
                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        }
        
        print("\nFinal generated questions:", questions)  # Debugging
        return list(questions.values())
    
    def generate_position_organization_questions_plural(self, graph, doc_num):
        """
        Who are both [Position1]s and [Position2]s of [Org Name]?
        """
        questions = []
        org_info = self.get_org_employees_positions(graph)
        
        for org_uri, info in org_info.items():
            org_name = self.clean_uri(org_uri)
            
            # Find positions with multiple holders
            multi_positions = [
                (self.clean_uri(pos), holders)
                for pos, holders in info['positions'].items()
                if len(holders) > 1
            ]
            
            # Only when exactly two such positions
            if len(multi_positions) == 2:
                # Unpack
                (pos1, holders1), (pos2, holders2) = sorted(multi_positions, key=lambda x: x[0])
                
                # Build question text
                # e.g. "Who are both Directors and Managers of AcmeCorp?"
                q_text = (
                    f"Who are both {pos1}s and {pos2}s of {org_name}?"
                )
                
                # Collect all holders from both positions
                all_holders = set()
                for h in holders1 | holders2:
                    all_holders.add(self.clean_uri(h))
                answer_text = ", ".join(sorted(all_holders))
                
                questions.append({
                    'question': q_text,
                    'answer': answer_text,
                    'num_hops': 2,               # org → position → people
                    'num_set_operations': 1,     # one union of two holder sets
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })
        
        return questions
    

    def generate_person_position_by_org_role_questions_plural(self, graph, doc_num):
        """
        Who are the [Person Positions] of the company which is the [Org Role (+ Sub‑Role)] in the agreement?
        """
        questions = []
        # 1) Flat org → {(role, subrole)}
        org_roles = self.get_org_roles_with_subroles(graph)

        # 2) Invert to normalized_role_string → { orgs }
        role_to_org = defaultdict(set)
        for org, pairs in org_roles.items():
            for base_role, subrole in pairs:
                norm = f"{subrole} {base_role}".strip() if subrole else base_role
                role_to_org[norm].add(org)

        # 3) Fetch once: org → positions → holders
        org_positions = self.get_org_employees_positions(graph)

        for norm_role, orgs in role_to_org.items():
            if len(orgs) != 1:
                continue
            org = next(iter(orgs))
            positions = org_positions.get(org, {}).get('positions', {})

            # Keep only positions held by exactly one person
            single_positions = {
                pos: holders
                for pos, holders in positions.items()
                if len(holders) == 1
            }
            if len(single_positions) < 2:
                continue

            # Clean & sort position names
            pos_list = sorted(self.clean_uri(pos) for pos in single_positions)
            if len(pos_list) == 2:
                pos_text = f"{pos_list[0]} and {pos_list[1]}"
            else:
                pos_text = ", ".join(pos_list[:-1]) + f", and {pos_list[-1]}"

            # Collect *all* holders across those positions
            all_holders = set()
            for holders in single_positions.values():
                all_holders.update(holders)

            # *** NEW FILTER: require >1 person in the answer ***
            if len(all_holders) <= 1:
                continue

            person_list = sorted(self.clean_uri(p) for p in all_holders)
            answer_text = ", ".join(person_list)

            q_text = (
                f"Who are the {pos_text} of the company which is the {norm_role} in the agreement?"
            )
            questions.append({
                'question': q_text,
                'answer': answer_text,
                'num_hops': 3,            
                'num_set_operations': 0,  
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })

        return questions

    
    def generate_location_position_questions_plural(self, graph, doc_num):
        """
        [Level 4]Who are the [Person Position]s of the company associated with [Location]?
        """
        questions = []
        loc_org_people = self.get_location_org_people_positions(graph)

        for location, orgs in loc_org_people.items():
            position_holders = defaultdict(set)
            for people in orgs.values():
                for person, positions in people.items():
                    for pos in positions:
                        position_holders[pos].add(person)

            # Only multi-holder positions
            for pos, holders in position_holders.items():
                if len(holders) > 1:
                    # simple pluralization by adding 's'
                    pos_pl = pos + 's'
                    ans = ", ".join(sorted(holders))
                    q = f"Who are the {pos_pl} of the company associated with {location}?"
                    questions.append({
                        'question': q,
                        'answer': ans,
                        'num_hops': 3,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1
                    })

        return questions
    
    def generate_employee_org_position_questions_plural(self, graph, doc_num):
        """
        Who are the [Position]s of the company where [Employee] is employed?
        """
        questions = []
        org_info = self.get_org_employees_positions(graph)
        
        for org_name, info in org_info.items():
            for employee in info['employees']:
                for position, holders in info['positions'].items():
                    # skip positions held by the employee themself
                    if employee in holders:
                        continue
                    # only multi-holder positions
                    if len(holders) > 1:
                        pos_pl = position + "s"
                        ans = ", ".join(sorted(holders))
                        q = f"Who are the {pos_pl} of the company where {employee} is employed?"
                        questions.append({
                            'question': q,
                            'answer': ans,
                            'num_hops': 3,
                            'num_set_operations': 0,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 1
                        })
        return questions
    
    def generate_dual_person_position_questions(self, graph, doc_num):
        """Who is both the [Person Position 1] and [Person Position 2] of the company which is the [Org Role(-s) (+ Sub-Role(-s))] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = {}

        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)

        # Get all organizations and their people with positions
        org_people_positions = self.get_org_people_positions(graph)

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)

        # Modify role-subrole pairs: replace "None" subroles with an empty string instead of removing the pair
        cleaned_org_roles = {
            org: {(role, subrole if subrole and subrole.lower() != "none" else "") for role, subrole in roles}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Identify companies with a **single, uniquely identifiable role**
        uniquely_identified_companies = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) == 1}

        # Keep track of already generated questions to remove duplicates
        seen_questions = set()

        # Process each uniquely identified company
        for org_name, roles in uniquely_identified_companies.items():
            if org_name in org_people_positions:  # Ensure the company has known employees
                # Iterate through each person in the organization
                for person, positions in org_people_positions[org_name].items():
                    if len(positions) == 2:
                        position_1, position_2 = sorted(positions)  # Ensure consistent ordering
                        role, subrole = list(roles)[0]  # Extract the unique role

                        # Format role-subrole pair properly, using empty string for None subroles
                        role_desc = f"{subrole} {role}" if subrole else role

                        # Construct question
                        question = (f"Who is both the {position_1} and {position_2} of the company "
                                    f"which is the {role_desc} in the agreement?")
                        
                        if question in seen_questions:
                            continue
                        
                        # Mark the question as seen
                        seen_questions.add(question)

                        # Answer is the person holding both positions
                        answer = person

                        print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                        # Store the question and answer
                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        }
        
        return list(questions.values())
    
    def generate_dual_person_position_location_questions(self, graph, doc_num):
        """Who is both the [Person Position 1] and [Person Position 2] of the company associated with [Location]? [if one]"""
        questions = {}

        # Get all organizations and their people with positions
        org_people_positions = self.get_org_people_positions(graph)
        print("Extracted organization people and positions:", org_people_positions)  # Debugging

        # Get all organizations and their associated locations
        location_org_data = self.get_org_roles_and_locations(graph)
        print("Extracted location-organization mapping:", location_org_data)  # Debugging

        # Keep track of already generated questions to remove duplicates
        seen_questions = set()

        # Process each location and its associated organizations
        for location, orgs in location_org_data.items():
            print(f"Processing location: {location} with organizations: {orgs}")  # Debugging

            for org_name in orgs:
                print(f"Checking organization: {org_name}")  # Debugging

                if org_name in org_people_positions:  # Ensure the company has known employees
                    for person, positions in org_people_positions[org_name].items():
                        print(f"Person: {person} has positions: {positions}")  # Debugging

                        if len(positions) == 2:
                            position_1, position_2 = sorted(positions)  # Ensure consistent ordering

                            # Construct question
                            question = (f"Who is both the {position_1} and {position_2} of the company "
                                        f"associated with {location}?")

                            # Skip if the question has already been generated
                            if question in seen_questions:
                                continue
                            
                            # Mark the question as seen
                            seen_questions.add(question)

                            # Answer is the single known position holder
                            answer = person

                            print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                            # Store the question and answer
                            questions[question] = {
                                'question': question,
                                'answer': answer,
                                'num_hops': 3,  
                                'num_set_operations': 1,
                                'document_number': doc_num,
                                'multiple_answer_dimension': 0
                            }
        
        print("Final generated questions:", questions)  # Debugging
        return list(questions.values())
    
    def generate_dual_position_questions(self, graph, doc_num):
        """Who is both the [Person Position 1] and [Person Position 2] of the company associated where [Person Name] is employed? [if one]"""
        questions = {}

        # Extract employment data: Organizations and their employees with positions
        org_people_positions = self.get_org_people_positions(graph)

        print("Extracted organization people and positions:", org_people_positions)  # Debugging

        # Keep track of already generated questions to remove duplicates
        seen_questions = set()

        # Iterate over each organization
        for org_name, employees in org_people_positions.items():
            print(f"Processing organization: {org_name}")  # Debugging

            # For each employee in the organization
            for person_name, positions in employees.items():
                # Only generate questions for people holding at least two distinct positions
                if len(positions) >= 2:
                    positions_list = list(positions)  # Convert the set to a list
                    print(f"Positions for {person_name}: {positions_list}")  # Debugging

                    # Find other employees who hold only one position
                    for other_person_name, other_positions in employees.items():
                        if len(other_positions) == 1 and other_person_name != person_name:
                            # Create the question
                            question = (f"Who is both the {positions_list[0]} and {positions_list[1]} "
                                        f"of the company where {other_person_name} is employed?")

                            # Skip if the question has already been generated
                            if question in seen_questions:
                                continue
                            
                            # Mark the question as seen
                            seen_questions.add(question)

                            # Store the question with the person as the answer
                            questions[question] = {
                                'question': question,
                                'answer': person_name,
                                'num_hops': 3,
                                'num_set_operations': 1,
                                'document_number': doc_num,
                                'multiple_answer_dimension': 0
                            }

        print("Final generated questions:", questions)  # Debugging
        return list(questions.values())
    
    def generate_location_dual_role_subrole_questions(self, graph, doc_num):
        """What is the address of the [Location Type] office of the company which is both the [Org Role(-s) (+ Sub-Role(-s)) 1] and the [Org Role(-s) (+ Sub-Role(-s)) 2] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = {}

        # Extract organization roles and subroles: Organizations and their roles/subroles
        org_roles_with_subroles = self.get_org_roles_with_subroles(graph)

        # Extract location-org data: Locations and their organizations with associated location types
        location_org_data = self.get_location_org_data(graph)

        print("Extracted location-org data:", location_org_data)  # Debugging

        # Normalize location type values (case-insensitive)
        def normalize_location_type(location_type):
            return location_type.strip().lower()

        # Filter organizations with more than one role-subrole pair
        organizations_with_multiple_roles = {}

        # Iterate over each location
        for location, org_info in location_org_data.items():
            print(f"Processing location: {location}")  # Debugging
            
            # For each organization in the location
            for org_name, details in org_info.items():
                location_types = details['location_types']
                
                # Normalize location types to handle case differences
                normalized_location_types = {normalize_location_type(loc) for loc in location_types}
                print(f"Normalized location types for {org_name}: {normalized_location_types}")  # Debugging

                # Check if the organization has roles or subroles
                roles_with_subroles = org_roles_with_subroles.get(org_name, [])
                print(f"Roles and subroles for {org_name}: {roles_with_subroles}")  # Debugging

                # If the organization has more than one role-subrole pair, add to filter
                if len(roles_with_subroles) > 1:
                    organizations_with_multiple_roles[org_name] = roles_with_subroles

        print("Organizations with more than one role-subrole pair:", organizations_with_multiple_roles)  # Debugging

        # Now generate questions based on these filtered organizations
        for org_name, roles_with_subroles in organizations_with_multiple_roles.items():
            location_info = location_org_data.get(org_name, {})
            
            # For each organization, generate a question based on the roles and location
            for location, org_info in location_org_data.items():
                for org_name, details in org_info.items():
                    location_types = details['location_types']
                    roles_list = []
                    for role, subrole in roles_with_subroles:
                            if subrole:
                                roles_list.append(f"{role} ({subrole})")
                            else:
                                roles_list.append(role)

                    # Normalize location types to handle case differences
                    normalized_location_types = {normalize_location_type(loc) for loc in location_types}
                    location_types_list = list(normalized_location_types)  # Convert the set to a list

                    # Generate a question based on location type, roles, and subroles
                    question = (f"What is the address of the {location_types_list[0]} office of the company "
                                    f"which is both the {roles_list[0]} and {roles_list[1]} in the agreement?")

                    # Store the question with the corresponding information
                    questions[question] = {
                            'question': question,
                            'organization': org_name,
                            'roles': roles_list,
                            'answer': location,
                            'num_hops': 3,  
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension':0
                        }

        print("Final generated questions:", questions)  
        return list(questions.values())

    def generate_all_questions(self):
        """Generate questions from all TTL files"""
        all_questions = []
        
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) 
                    if not f.endswith("ontology.ttl")]
        
        for ttl_file in ttl_files:
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                graph = self.load_graph(ttl_file)

                all_questions.extend(self.generate_position_comparison_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_org_role_comparison_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_org_role_subrole_comparison_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_position_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_shared_position_exclusion_questions(graph, doc_num))
                all_questions.extend(self.generate_exclusive_shared_role_questions(graph, doc_num))
                all_questions.extend(self.generate_exclusive_role_subrole_questions(graph, doc_num))
                all_questions.extend(self.generate_shared_role_subrole_exclusive_questions(graph, doc_num))
                all_questions.extend(self.generate_org_role_exclusion_questions(graph, doc_num))
                all_questions.extend(self.generate_position_organization_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_person_position_by_org_role_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_location_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_employee_org_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_person_position_questions_for_unique_dual_role(graph, doc_num))
                all_questions.extend(self.generate_dual_person_position_questions(graph, doc_num))
                all_questions.extend(self.generate_dual_person_position_location_questions(graph, doc_num))
                all_questions.extend(self.generate_dual_position_questions(graph, doc_num))
                all_questions.extend(self.generate_location_dual_role_subrole_questions(graph, doc_num))
                
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
        df = pd.DataFrame(all_questions, columns = columns_order)
        df.fillna("")
        
        return df

def main():
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level4.csv', index=False)
    print(f"Generated {len(df)} questions and saved to level4.csv")

if __name__ == "__main__":
    main()
