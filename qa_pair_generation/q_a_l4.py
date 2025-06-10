import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from itertools import combinations
from collections import defaultdict
from typing import Dict, Set
from urllib.parse import unquote


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
    def get_org_roles_and_locations(self, graph):
        """Get all organizations, their roles (without subroles), and locations from the graph."""
        org_data = {}

        # Simplified query without subrole resolution
        query = """
            SELECT DISTINCT ?org ?location (GROUP_CONCAT(DISTINCT ?role; separator="|") as ?roles)
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                    <http://example.org/isInstanceOf/> ?role ;
                    <http://example.org/relation/hasLocationAt> ?location .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }
            GROUP BY ?org ?location
        """

        results = graph.query(query)

        for row in results:
            org_uri = row[0]
            location_uri = row[1]
            role_uris = row[2].split('|') if row[2] else []

            # Clean URIs
            org_name = self.clean_uri(org_uri)
            location = self.clean_uri(location_uri)
            roles = set(self.clean_uri(role_uri) for role_uri in role_uris)

            # Initialize location entry if needed
            if location not in org_data:
                org_data[location] = {}
            
            # Add organization and its roles to location
            org_data[location][org_name] = roles

        return org_data
    
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
        org_roles = self.get_org_roles(graph)  

        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]

            # at least one role in common
            if not (roles1 & roles2):
                continue

            # find all roles in org1 but not in org2
            unique_roles = sorted(roles1 - roles2)
            if len(unique_roles) > 1:
                answer = ", ".join(unique_roles)
                questions.append({
                    'question': (
                        f"What roles does {org1} have in the agreement "
                        f"which are not the roles of {org2}?"
                    ),
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 2,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })


            unique_roles_rev = sorted(roles2 - roles1)
            if len(unique_roles_rev) > 1:
                answer_rev = ", ".join(unique_roles_rev)
                questions.append({
                    'question': (
                        f"What roles does {org2} have in the agreement "
                        f"which are not the roles of {org1}?"
                    ),
                    'answer': answer_rev,
                    'num_hops': 1,
                    'num_set_operations': 2,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions
    

    def generate_org_role_subrole_comparison_questions_plural(self, graph, doc_num):
        """What companies are the [Role 1] but not the [Role 2] in the agreement?"""
        questions = []
        role_org_map = defaultdict(set)
        
        # Get organizations and their roles
        org_roles = self.get_org_roles(graph)  # {org: set(role_strings)}
        
        # Build role->organizations mapping
        for org_name, roles in org_roles.items():
            for role in roles:
                role_org_map[role].add(org_name)
        
        # Process all role pairs
        for role1, role2 in combinations(role_org_map.keys(), 2):
            orgs1 = role_org_map[role1]
            orgs2 = role_org_map[role2]
            
            # Ensure at least one organization has both roles
            if not (orgs1 & orgs2):
                continue
            
            # Generate questions for both directions
            for role_a, role_b in [(role1, role2), (role2, role1)]:
                orgs_a = role_org_map[role_a]
                orgs_b = role_org_map[role_b]
                unique_orgs = sorted(orgs_a - orgs_b)
                
                if unique_orgs:
                    question = (
                        f"What companies are the {role_a} "
                        f"but not the {role_b} in the agreement?"
                    )
                    questions.append({
                        'question': question,
                        'answer': ", ".join(unique_orgs),
                        'num_hops': 1,
                        'num_set_operations': 2,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1 if len(unique_orgs) > 1 else 0
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
        """What role do [Org Name 1] and [Org Name 2] share in the agreement that is not held by [Org Name 3]? [if one]"""
        questions = []
        seen_questions = set()

        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # {org: set(role_strings)}
        
        # Consider all organizations with at least one role
        valid_orgs = {org: roles for org, roles in org_roles.items() if roles}
        
        # Generate all unique combinations of three organizations
        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]
            
            # Check that all three orgs share at least one role
            common_roles_all_three = roles1 & roles2 & roles3
            if not common_roles_all_three:
                continue  # Skip if no role is shared by all three
            
            # Find roles shared by Org1 and Org2
            shared_12 = roles1 & roles2
            if not shared_12:
                continue  # Skip if no shared roles
                
            # Find roles exclusive to Org1 and Org2 (not held by Org3)
            exclusive_shared_roles = shared_12 - roles3
            if not exclusive_shared_roles:
                continue  # Skip if no exclusive shared roles
                
            # Create sorted role list for consistent formatting
            sorted_roles = sorted(exclusive_shared_roles)
            role_str = ", ".join(sorted_roles)
            
            # Create unique key for question deduplication
            key = (frozenset([org1, org2]), frozenset(exclusive_shared_roles), org3)
            if key in seen_questions:
                continue
            seen_questions.add(key)
            
            # Generate question
            question = (f"What role do {org1} and {org2} share in the agreement "
                        f"that is not held by {org3}?")
            
            # Determine answer dimension (single vs multiple)
            multiple_dim = 1 if len(exclusive_shared_roles) > 1 else 0
            
            questions.append({
                'question': question,
                'answer': role_str,
                'num_hops': 1,
                'num_set_operations': 3,
                'document_number': doc_num,
                'multiple_answer_dimension': multiple_dim
            })
        
        return questions

    
    def generate_exclusive_role_subrole_questions(self, graph, doc_num):
        """
        What role does [Org Name 1] have in the agreement which is not the role of [Org Name 2] or [Org Name 3]? [if one]
        """
        questions = []
        seen = set()

        # 1) Fetch org → { "subrole role" strings }
        org_roles = self.get_org_roles(graph)

        # 2) Keep only orgs with at least two roles
        valid_orgs = {org: roles for org, roles in org_roles.items() if len(roles) > 1}

        # 3) Iterate all triples
        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]

            # Must share at least one role with both peers
            if not (roles1 & roles2) or not (roles1 & roles3):
                continue

            # Exclusive roles = in 1 but not in 2 or 3
            exclusive = roles1 - (roles2 | roles3)
            if len(exclusive) != 1:
                continue

            role = next(iter(exclusive))
            key = (org1, role, frozenset([org2, org3]))
            if key in seen:
                continue
            seen.add(key)

            question = (
                f"What role does {org1} have in the agreement which is not the role of "
                f"{org2} or {org3}?"
            )
            questions.append({
                'question': question,
                'answer': role,
                'num_hops': 1,
                'num_set_operations': 3,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })

        return questions
    
    def generate_shared_role_subrole_exclusive_questions(self, graph, doc_num):
        """What company is the [Role1] and [Role2] but not [Role3] in the agreement?."""
        questions = []


        org_roles = self.get_org_roles(graph)
        role_to_orgs = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role_str in roles:
                role_to_orgs[role_str].add(org_name)
        for r1, r2, r3 in combinations(role_to_orgs.keys(), 3):
            orgs1 = role_to_orgs[r1]
            orgs2 = role_to_orgs[r2]
            orgs3 = role_to_orgs[r3]
            if not (orgs1 & orgs3) and (orgs2 & orgs3):
                continue
            orgs12 = orgs1 & orgs2
            final_orgs = orgs12 - orgs3
            if len(final_orgs) == 1:
                only_org = next(iter(final_orgs))
                q = (
                    f"What company is the {r1} and {r2} "
                    f"but not the {r3} in the agreement?"
                )
                questions.append({
                    'question': q,
                    'answer': only_org,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions
    


    def generate_org_role_exclusion_questions(self, graph, doc_num):
        """
        What company is the [Role1] but not the [Role2] and [Role3] in the agreement?
        Only if all three roles co-occur somewhere and exactly one company has Role1 without Role2 or Role3.
        """
        questions = []

        
        org_roles = self.get_org_roles(graph)  
        role_to_orgs = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_orgs[role].add(org_name)

        for r1, r2, r3 in combinations(role_to_orgs.keys(), 3):
            orgs1, orgs2, orgs3 = role_to_orgs[r1], role_to_orgs[r2], role_to_orgs[r3]
            if not (orgs1 & orgs2 & orgs3):
                continue

            # Companies that have r1 but neither r2 nor r3
            exclusive = orgs1 - (orgs2 | orgs3)
            if len(exclusive) == 1:
                only_org = next(iter(exclusive))
                q_text = (
                    f"What company is the {r1} "
                    f"but not the {r2} and {r3} in the agreement?"
                )
                questions.append({
                    'question': q_text,
                    'answer': only_org,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions

    def generate_person_position_questions_for_unique_dual_role(self, graph, doc_num):
        "Who is the {position} of the company which is both the [Role1] and the [Role2] in the agreement?"
        questions = {}  
        
        # Get organizations with employees/positions
        org_info = self.get_org_employee_positions(graph)
        
        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # Returns {org: set(role_strings)}
        
        # Find companies with exactly two distinct roles
        dual_role_companies = {}
        for org, roles in org_roles.items():
            if len(roles) == 2:  
                dual_role_companies[org] = roles

        seen_questions = set()  # Track duplicates

        for org_name, roles in dual_role_companies.items():
            if org_name in org_info:  # Only companies with position data
                for position, holders in org_info[org_name]['positions'].items():
                    if len(holders) == 1:  # Unique position holder
                        # Sort roles for consistent question phrasing
                        sorted_roles = sorted(roles)
                        role1 = sorted_roles[0]
                        role2 = sorted_roles[1]
                        
                        # Generate question using roles only (no subroles)
                        question = (f"Who is the {position} of the company which is both the {role1} "
                                    f"and the {role2} in the agreement?")
                        
                        if question in seen_questions:
                            continue  # Skip duplicates
                        seen_questions.add(question)
                        
                        answer = list(holders)[0]  # Single holder
                        
                        # Store question metadata
                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        }
        return list(questions.values())
    
    def generate_position_organization_questions_plural(self, graph, doc_num):
        """
        Who are both [Position1]s and [Position2]s of [Org Name]?
        """
        questions = []
        org_info = self.get_org_employee_positions(graph)
        
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
                    'num_hops': 2,               
                    'num_set_operations': 1,    
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })
        
        return questions
    


    def generate_person_position_by_org_role_questions_plural(self, graph, doc_num):
        """
        Who are the [Person Positions] of the company which is the [Org Role] in the agreement?

        """
        questions = []

        # 1) org → { role_str }
        org_roles = self.get_org_roles(graph)  # { org_name: set("Treasurer", "CEO", ...) }

        # 2) invert into role_str → { org_name }
        role_to_org = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_org[role].add(org_name)

        # 3) prefetch org → { positions: {pos_str: {people}} }
        org_positions = self.get_org_employee_positions(graph)

        for role, orgs in role_to_org.items():
            # only consider roles held by exactly one company
            if len(orgs) != 1:
                continue
            the_org = next(iter(orgs))

            # fetch that company’s positions
            positions_dict = org_positions.get(the_org, {}).get('positions', {})
            # keep only positions held by exactly one person
            singleton_positions = {
                pos: holders
                for pos, holders in positions_dict.items()
                if len(holders) == 1
            }
            # need at least two such positions to ask a plural “Who are the A and B…”
            if len(singleton_positions) < 2:
                continue

            # build a human‑readable list of position names
            pos_names = sorted(pos for pos in singleton_positions.keys())
            if len(pos_names) == 2:
                pos_text = f"{pos_names[0]} and {pos_names[1]}"
            else:
                pos_text = ", ".join(pos_names[:-1]) + f", and {pos_names[-1]}"

            # collect all the people holding them
            holders = set()
            for hset in singleton_positions.values():
                holders.update(hset)
            # require >1 distinct person
            if len(holders) <= 1:
                continue

            person_names = sorted(holders)
            answer_text = ", ".join(person_names)

            questions.append({
                'question': (
                    f"Who are the {pos_text} of the company which is the {role} in the agreement?"
                ),
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
        org_info = self.get_org_employee_positions(graph)
        
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
        """Who is both the [Person Position 1] and [Person Position 2] of the company which is the [Org Role] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = {}

        # Get organizations with people and their positions
        org_people_positions = self.get_org_people_positions(graph)
        
        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # Returns {org: set(role_strings)}

        # Identify companies with exactly one role
        single_role_companies = {org: roles for org, roles in org_roles.items() if len(roles) == 1}

        seen_questions = set()  # Track duplicates

        for org_name, roles in single_role_companies.items():
            if org_name in org_people_positions:  # Only companies with position data
                # Extract the single role for this company
                role = next(iter(roles))
                
                # Process each person in the company
                for person, positions in org_people_positions[org_name].items():
                    if len(positions) == 2:  # Person holds exactly two positions
                        # Sort positions for consistent question phrasing
                        sorted_positions = sorted(positions)
                        position1 = sorted_positions[0]
                        position2 = sorted_positions[1]
                        
                        # Generate question
                        question = (f"Who is both the {position1} and {position2} of the company "
                                    f"which is the {role} in the agreement?")
                        
                        if question in seen_questions:
                            continue  # Skip duplicates
                        seen_questions.add(question)
                        
                        # Store question metadata
                        questions[question] = {
                            'question': question,
                            'answer': person,
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
        """What is the address of the [Location Type] office of the company which is both the [Org Role 1] and the [Org Role 2] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = {}

        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # {org: set(role_strings)}
        
        # Get location data: {location: {org: {'location_types': set}}}
        location_org_data = self.get_location_org_data(graph)
        
        # Create organization location index: {org: {location: location_types}}
        org_location_index = {}
        for location, orgs in location_org_data.items():
            for org, details in orgs.items():
                if org not in org_location_index:
                    org_location_index[org] = {}
                org_location_index[org][location] = details['location_types']

        # Find organizations with exactly two roles
        dual_role_orgs = {org: roles for org, roles in org_roles.items() if len(roles) == 2}

        seen_questions = set()
        for org, roles in dual_role_orgs.items():
            if org in org_location_index:
                # Get sorted roles for consistent question phrasing
                sorted_roles = sorted(roles)
                role1, role2 = sorted_roles
                
                # Process each location for this organization
                for location, location_types in org_location_index[org].items():
                    # Normalize and get primary location type
                    normalized_types = sorted({lt.strip().lower() for lt in location_types})
                    if not normalized_types:
                        continue
                    primary_type = normalized_types[0]
                    
                    # Generate question
                    question = (f"What is the address of the {primary_type} office of the company "
                                f"which is both the {role1} and {role2} in the agreement?")
                    
                    if question in seen_questions:
                        continue
                    seen_questions.add(question)
                    
                    # Store question metadata
                    questions[question] = {
                        'question': question,
                        'organization': org,
                        'roles': sorted_roles,
                        'answer': location,
                        'num_hops': 3,
                        'num_set_operations': 1,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    }
        return list(questions.values())

    def generate_all_questions(self):
        """Generate questions from all TTL files"""
        all_questions = []
        
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) 
                    if not f.endswith("ontology.ttl")]
        
        for ttl_file in ttl_files:
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                graph = self.load_graph(ttl_file)

                # all_questions.extend(self.generate_position_comparison_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_org_role_comparison_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_org_role_subrole_comparison_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_position_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_shared_position_exclusion_questions(graph, doc_num))
                # all_questions.extend(self.generate_exclusive_shared_role_questions(graph, doc_num))
                # all_questions.extend(self.generate_exclusive_role_subrole_questions(graph, doc_num))
                # all_questions.extend(self.generate_shared_role_subrole_exclusive_questions(graph, doc_num))
                # all_questions.extend(self.generate_org_role_exclusion_questions(graph, doc_num))
                # all_questions.extend(self.generate_position_organization_questions_plural(graph, doc_num)) # no answers
                # all_questions.extend(self.generate_person_position_by_org_role_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_location_position_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_employee_org_position_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_person_position_questions_for_unique_dual_role(graph, doc_num))
                # all_questions.extend(self.generate_dual_person_position_questions(graph, doc_num))
                # all_questions.extend(self.generate_dual_person_position_location_questions(graph, doc_num))
                # all_questions.extend(self.generate_dual_position_questions(graph, doc_num))
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
