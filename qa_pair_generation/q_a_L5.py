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


    def _is_duplicate_question(self, questions, new_question, new_answer):
        """Helper function to check if a question-answer pair is an exact duplicate"""
        for q in questions:
            if new_question == q['question'] and new_answer == q['answer']:
                return True
        return False
    
    def generate_multiple_position_comparison_questions_plural(self, graph, doc_num):
        """
        [Level 5] What are the positions held by [Person Name 1] but not by [Person Name 2] or [Person Name 3]?
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

            if len(unique_positions) > 0:
                # Convert unique positions to a sorted list
                unique_positions_list = sorted(unique_positions)

                if len(unique_positions_list)>1:
                    question = f"What are the positions held by {person1} but not by {person2} or {person3}?"
                    answer = ", ".join(unique_positions_list)

                    print(f"Generated question: {question} | Answer: {answer}")

                    questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1, 
                    'num_set_operations': 3, 
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                    })

        print("\nFinal generated questions:", questions)
        return questions
    
    def generate_position_comparison_questions_plural(self, graph, doc_num):
        """
        What are the positions held by Person1 and Person2 but not by Person3?
        (Only when at least two such positions exist.)
        """
        questions = []
        seen = set()
        pp = self.get_person_positions(graph)

        for p1, p2, p3 in combinations(pp, 3):
            P1, P2, P3 = pp[p1], pp[p2], pp[p3]

            shared_by_two = (P1 & P2) - P3
            if len(shared_by_two) <= 1:
                continue

            key = tuple(sorted([p1, p2, p3])) + tuple(sorted(shared_by_two))
            if key in seen:
                continue
            seen.add(key)

            question = f"What are the positions held by {p1} and {p2} but not by {p3}?"
            answer   = ", ".join(sorted(shared_by_two))
            questions.append({
                'question': question,
                'answer':   answer,
                'num_hops': 1,
                'num_set_operations': 3,  
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        return questions
    def generate_exclusive_shared_role_questions_plural(self, graph, doc_num):
        """[Level 5] What roles do [Org Name 1] and [Org Name 2] share in the agreement that are not held by [Org Name 3]?"""
        questions = []
        seen_questions = set()

        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # {org: set(role_strings)}
        
        # Consider all organizations with roles
        valid_orgs = {org: roles for org, roles in org_roles.items() if roles}
        
        # Generate all unique combinations of three organizations
        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]
            
            # Check shared roles with org3
            if not (roles1 & roles3) or not (roles2 & roles3):
                continue  # Skip if org1/org3 or org2/org3 have no roles in common

            # Find roles shared by Org1 and Org2
            shared_12 = roles1 & roles2
            if not shared_12:
                continue
                
            # Find roles exclusive to Org1 and Org2 (not held by Org3)
            exclusive_shared_roles = shared_12 - roles3
            if len(exclusive_shared_roles) < 2:  # Require at least 2 roles
                continue
                
            # Create sorted role list for consistent formatting
            sorted_roles = sorted(exclusive_shared_roles)
            role_str = ", ".join(sorted_roles)
            
            # Create unique key for question deduplication
            key = (frozenset([org1, org2]), frozenset(exclusive_shared_roles), org3)
            if key in seen_questions:
                continue
            seen_questions.add(key)
            
            # Generate question
            question = (f"What roles do {org1} and {org2} share in the agreement "
                        f"that are not held by {org3}?")
            
            questions.append({
                'question': question,
                'answer': role_str,
                'num_hops': 1,
                'num_set_operations': 3,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })
        
        return questions
    
    def generate_exclusive_role_subrole_questions_plural(self, graph, doc_num):
        """
        What roles does [Org Name 1] have in the agreement which are not the roles of [Org Name 2] or [Org Name 3]? [if plural]
        """
        questions = []
        seen = set()

        # 1) Fetch org → set of role strings
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
            if len(exclusive) <= 1:
                continue

            key = (org1, frozenset(exclusive), frozenset([org2, org3]))
            if key in seen:
                continue
            seen.add(key)

            sorted_roles = sorted(exclusive)
            if len(sorted_roles)>1:
                answer = ", ".join(sorted_roles)

            question = (
                f"What roles does {org1} have in the agreement which are not the roles of "
                f"{org2} or {org3}?"
            )
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 3,
                'document_number': doc_num,
                'multiple_answer_dimension': 1
            })

        return questions
    
    def generate_shared_role_subrole_exclusive_questions_plural(self, graph, doc_num):
        """
        What companies are the [Role1] and [Role2] but not [Role3] in the agreement? (plural)
        """
        questions = []

        # Fetch org → set of roles
        org_roles = self.get_org_roles(graph)

        # Build inverse map: role → set of orgs
        role_to_orgs = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role_str in roles:
                role_to_orgs[role_str].add(org_name)

        for r1, r2, r3 in combinations(role_to_orgs.keys(), 3):
            orgs1 = role_to_orgs[r1]
            orgs2 = role_to_orgs[r2]
            orgs3 = role_to_orgs[r3]

            # Require meaningful overlap (avoid degenerate case)
            if not (orgs1 & orgs2):
                continue
            if not(orgs1 & orgs3):
                continue

            orgs12 = orgs1 & orgs2
            final_orgs = orgs12 - orgs3

            if len(final_orgs) > 1:
                sorted_orgs = sorted(final_orgs)
                answer = ", ".join(sorted_orgs[:-1]) + f", and {sorted_orgs[-1]}" if len(sorted_orgs) > 2 else " and ".join(sorted_orgs)

                q = (
                    f"What companies are the {r1} and {r2} "
                    f"but not the {r3} in the agreement?"
                )
                questions.append({
                    'question': q,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions

    def generate_org_role_exclusion_questions_plural(self, graph, doc_num):
        """
        What companies are the [Role1] but not the [Role2] and [Role3] in the agreement?
        """
        questions = []

        org_roles = self.get_org_roles(graph)  
        role_to_orgs = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_orgs[role].add(org_name)

        for r1, r2, r3 in combinations(role_to_orgs.keys(), 3):
            orgs1, orgs2, orgs3 = role_to_orgs[r1], role_to_orgs[r2], role_to_orgs[r3]

            # Ensure the three roles co-occur in the data (i.e., some overlap)
            if not (orgs1 & orgs2 & orgs3):
                continue

            # Companies that have r1 but neither r2 nor r3
            exclusive = orgs1 - (orgs2 | orgs3)
            if len(exclusive) > 1:
                q_text = (
                    f"What companies are the {r1} "
                    f"but not the {r2} and {r3} in the agreement?"
                )
                answer = ", ".join(sorted(exclusive))
                questions.append({
                    'question': q_text,
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions
    
    def generate_person_position_questions_for_unique_dual_role_plural(self, graph, doc_num):
        """
        Who are the [Position]s of the company which is both the [Role1] and the [Role2] in the agreement?
        """
        questions = {}  

        # Get organizations with employees/positions
        org_info = self.get_org_employees_positions(graph)

        # Get organizations and their roles (without subroles)
        org_roles = self.get_org_roles(graph)  # Returns {org: set(role_strings)}

        # Find companies with exactly two distinct roles
        dual_role_companies = {
            org: roles for org, roles in org_roles.items() if len(roles) == 2
        }

        seen_questions = set()

        for org_name, roles in dual_role_companies.items():
            if org_name in org_info:  # Must have position data
                for position, holders in org_info[org_name]['positions'].items():
                    if len(holders) > 1:  # Plural case
                        sorted_roles = sorted(roles)
                        role1 = sorted_roles[0]
                        role2 = sorted_roles[1]

                        question = (
                            f"Who are the {position}s of the company which is both the {role1} "
                            f"and the {role2} in the agreement?"
                        )

                        if question in seen_questions:
                            continue
                        seen_questions.add(question)

                        answer = ", ".join(sorted(holders))

                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 1
                        }

        return list(questions.values())

    def generate_dual_person_position_questions_plural(self, graph, doc_num):
        """
        Who are both the [Position1]s and [Position2]s of [Org Name]?
        """
        questions = []
        seen = set()

        # 1) Fetch org → { person → positions }
        org_people = self.get_org_people_positions(graph)

        for org_uri, people_map in org_people.items():
            org_name = self.clean_uri(org_uri)
            # Map each exact two-position tuple → list of people
            combo_to_people = defaultdict(list)
            for person, positions in people_map.items():
                if len(positions) == 2:
                    combo = tuple(sorted(positions))
                    combo_to_people[combo].append(person)

            # Emit when >1 person shares the same two-position combo
            for (pos1, pos2), holders in combo_to_people.items():
                if len(holders) > 1:
                    # Clean position labels
                    p1 = self.clean_uri(pos1)
                    p2 = self.clean_uri(pos2)
                    
                    q_text = f"Who are both the {p1}s and {p2}s of {org_name}?"
                    key = (org_name, p1, p2, tuple(sorted(holders)))
                    if key in seen:
                        continue
                    seen.add(key)

                    # Clean person names
                    answer = ", ".join(sorted(self.clean_uri(h) for h in holders))
                    questions.append({
                        'question': q_text,
                        'answer': answer,
                        'num_hops': 3,               # org → person → positions
                        'num_set_operations': 1,     # grouping by combo
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1
                    })

        return questions
    
    def generate_dual_person_position_location_questions_plural(self, graph, doc_num):
        """
        Who are both the [Position1]s and [Position2]s of the company associated with [Location]?
        (Only when >1 people hold that exact two-position combination at that location.)
        """
        questions = []
        seen = set()

        # org_name → { person_name → set(position_str) }
        org_people_positions = self.get_org_people_positions(graph)
        print("ORG → PEOPLE → POSITIONS map:", org_people_positions)

        # location_str → [org_name_str, …]
        location_org_data = self.get_org_roles_and_locations(graph)
        print("LOCATION → ORGS map:", location_org_data)

        for location, orgs in location_org_data.items():
            print(f"\n--- Location: {location} (orgs: {orgs})")
            for org_name in orgs:
                print(f"Checking org: {org_name}")
                people_map = org_people_positions.get(org_name, {})
                print(f"  People at {org_name}: {list(people_map.keys())}")

                # Map each 2-position tuple to list of people
                combo_to_people = defaultdict(list)
                for person, positions in people_map.items():
                    print(f"    Person: {person} holds positions: {positions}")
                    if len(positions) == 2:
                        pos1, pos2 = sorted(positions)
                        combo_to_people[(pos1, pos2)].append(person)

                print(f"  Combos at {org_name}: {dict(combo_to_people)}")

                for (pos1, pos2), holders in combo_to_people.items():
                    print(f"    Combo ({pos1}, {pos2}) → holders: {holders}")
                    # only plural answers
                    if len(holders) > 1:
                        q_text = (
                            f"Who are both the {pos1}s and {pos2}s "
                            f"of the company associated with {location}?"
                        )
                        answer = ", ".join(sorted(holders))
                        key = (location, pos1, pos2, tuple(sorted(holders)))
                        if key in seen:
                            print("      Already seen, skipping.")
                            continue
                        seen.add(key)
                        print(f"      Generated question: {q_text} | Answer: {answer}")

                        questions.append({
                            'question': q_text,
                            'answer': answer,
                            'num_hops': 3,
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 1
                        })

        print("\nFinal questions generated:", questions)
        return questions
    
    def generate_dual_position_questions_plural(self, graph, doc_num):
        """
        Who are both the [Position1]s and [Position2]s of the company where [Person Name] is employed?
        (Only when more than one employee holds exactly those two positions at that company,
        and there is at least one other employee with exactly one position to reference.)
        """
        questions = []
        seen = set()

        # org_name_uri → { person_name → set(position_uris) }
        org_people_positions = self.get_org_people_positions(graph)
        print("=== Org -> Employees -> Positions Map ===")
        print(org_people_positions)

        for org_uri, employees in org_people_positions.items():
            org_name = self.clean_uri(org_uri)
            print("\n" + "="*40)
            print(f"Processing organization: {org_name}")
            print("Employees and their positions:")
            for person, positions in employees.items():
                print(f"  {person}: {positions}")

            # Build mapping: exact 2-position combos -> list of holders
            combo_to_people = defaultdict(list)
            for person, positions in employees.items():
                if len(positions) == 2:
                    combo = tuple(sorted(positions))
                    combo_to_people[combo].append(person)
                else:
                    print(f"  Skipping {person}: holds {len(positions)} position(s)")

            print("Combo -> Holders mapping:")
            for combo, holders in combo_to_people.items():
                p1, p2 = combo
                print(f"  {combo}: {holders}")

            # Identify single-position employees for reference
            single_position_people = [p for p, pos in employees.items() if len(pos) == 1]
            print("Single-position employees:", single_position_people)

            # Generate questions for combos with multiple holders
            for (pos1_uri, pos2_uri), holders in combo_to_people.items():
                print(f"\nEvaluating combo ({pos1_uri}, {pos2_uri}) with holders {holders}")
                if len(holders) <= 1:
                    print("  Skipping combo: not enough holders for plural question")
                    continue
                if not single_position_people:
                    print("  Skipping combo: no single-position employee available for reference")
                    continue

                ref_person = single_position_people[0]
                print(f"  Using reference person: {ref_person}")

                p1 = self.clean_uri(pos1_uri)
                p2 = self.clean_uri(pos2_uri)
                q_text = f"Who are both the {p1}s and {p2}s of {org_name} where {ref_person} is employed?"
                answer = ", ".join(sorted(self.clean_uri(h) for h in holders))
                print(f"  Generated question: {q_text} | Answer: {answer}")

                key = (org_name, tuple(sorted((p1, p2))), ref_person, tuple(sorted(holders)))
                if key in seen:
                    print("  Duplicate question detected, skipping")
                    continue
                seen.add(key)

                questions.append({
                    'question': q_text,
                    'answer': answer,
                    'num_hops': 3,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        print("\n=== Final generated questions ===")
        print(questions)
        return questions
    
    def generate_position_by_org_role_comparison_questions(self, graph, doc_num):
        """
        Who is the [Person Position] of the company which is the [Org Role 1]
        but not the [Org Role 2] in the agreement?
        """
        questions = []

        org_roles = self.get_org_roles(graph)  # {org: set of roles}
        org_info = self.get_org_employees_positions(graph)  # {org: {'positions': {position: set of people}}}

        orgs = list(org_roles.keys())

        for i, org1 in enumerate(orgs):
            roles1 = org_roles[org1]
            for j in range(len(orgs)):
                if i == j:
                    continue
                org2 = orgs[j]
                roles2 = org_roles[org2]

                shared_roles = roles1 & roles2
                if not shared_roles:
                    continue

                for excl_role in roles2:
                    if excl_role in roles1:
                        continue  

                    for shared_role in shared_roles:
                        if org1 not in org_info:
                            continue
                        for position, people in org_info[org1]['positions'].items():
                            if not people:
                                continue

                            question = (
                                f"Who is the {position} of the company which is the {shared_role} "
                                f"but not the {excl_role} in the agreement?"
                            )
                            answer = ", ".join(sorted(people))

                            if not self._is_duplicate_question(questions, question, answer):
                                questions.append({
                                    'question': question,
                                    'answer': answer,
                                    'num_hops': 3,
                                    'num_set_operations': 2,
                                    'document_number': doc_num,
                                    'multiple_answer_dimension': 0
                                })

        return questions


    
    def generate_position_comparison_by_org_role_questions(self, graph, doc_num):
        """Who is the [Person Position 1] but not [Person Position 2] of the company which is the [Organization Role] in the agreement"""
        questions = []

        query = """
        SELECT DISTINCT ?org ?role ?sub_role ?person ?position
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                <http://example.org/isInstanceOf/> ?position .
            {
                ?org <http://example.org/isInstanceOf/> ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }
            OPTIONAL {
                ?org <http://example.org/isInstanceOf/> ?sub_role .
                ?sub_role rdfs:subClassOf ?role .
                FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
            }
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        """

        results = graph.query(query)

        # org -> {roles: set(), role_subroles: set(), people: {person: set(positions)}, roles_with_subroles: set()}
        org_info = {}

        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            sub_role_uri = row[2] if row[2] else None
            person_uri = row[3]
            position_uri = row[4]

            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)

            if org_name not in org_info:
                org_info[org_name] = {
                    'roles': set(),
                    'role_subroles': set(),
                    'people': {},
                    'roles_with_subroles': set()
                }

            org_info[org_name]['roles'].add(role)
            if sub_role:
                org_info[org_name]['role_subroles'].add((role, sub_role))
                org_info[org_name]['roles_with_subroles'].add(role)

            if person_name not in org_info[org_name]['people']:
                org_info[org_name]['people'][person_name] = set()
            org_info[org_name]['people'][person_name].add(position)

        for org_name, info in org_info.items():
            role_descriptions = []

            for role, sub_role in info['role_subroles']:
                role_descriptions.append(f"{sub_role} {role}")
            for role in info['roles']:
                if role not in info['roles_with_subroles']:
                    role_descriptions.append(role)

            all_people = info['people']
            all_positions = set.union(*all_people.values()) if all_people else set()

            for role_str in role_descriptions:
                for person_name, person_positions in all_people.items():
                    for pos1 in person_positions:
                        for pos2 in all_positions:
                            if pos1 == pos2 or pos2 in person_positions:
                                continue  # Skip if person has both

                            # Generate question
                            question = f"Who is the {pos1} but not {pos2} of the company which is the {role_str} in the agreement?"
                            answer = person_name

                            if not self._is_duplicate_question(questions, question, answer):
                                questions.append({
                                    'question': question,
                                    'answer': answer,
                                    'num_hops': 3,
                                    'num_set_operations': 2,
                                    'document_number': doc_num,
                                    'multiple_answer_dimension': 0
                                })

        return questions



    def generate_location_position_comparison_questions(self, graph, doc_num):
        """
        "Who is the [Position1] but not [Position2] of the company associated with [Location]?"
        """
        questions = []

        # Structure: {location: {org: {person: set(positions)}}}
        location_org_people = self.get_location_org_people_positions(graph)

        for location, orgs in location_org_people.items():
            for org_name, people in orgs.items():
                for person, positions in people.items():
                    if len(positions) < 2:
                        continue  # Need at least 2 positions to compare

                    for pos1 in positions:
                        for pos2 in positions.union(set.union(*[p for p in people.values()])):
                            if pos1 == pos2 or pos2 in positions:
                                continue  # Person must NOT hold pos2

                            question = f"Who is the {pos1} but not {pos2} of the company associated with {location}?"
                            answer = person

                            if not self._is_duplicate_question(questions, question, answer):
                                questions.append({
                                    'question': question,
                                    'answer': answer,
                                    'num_hops': 3,
                                    'num_set_operations': 2,
                                    'document_number': doc_num,
                                    'multiple_answer_dimension': 0
                                })

        return questions

    def generate_position_comparison_by_employee_questions(self, graph, doc_num):
        """Who is the [Position 1] but not [Position 2] of the company where [Person] is employed?"""
        questions = []

        # SPARQL query to get organizations, their employees and their positions
        query = """
        SELECT DISTINCT ?org ?person ?position
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        """

        results = graph.query(query)

        # Map organizations to their employees and positions
        org_employee_positions = {}

        for row in results:
            org_uri = row[0]
            person_uri = row[1]
            position_uri = row[2]

            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)

            if org_name not in org_employee_positions:
                org_employee_positions[org_name] = {}

            if person_name not in org_employee_positions[org_name]:
                org_employee_positions[org_name][person_name] = set()

            org_employee_positions[org_name][person_name].add(position)

        for org_name, employees in org_employee_positions.items():
            person_names = list(employees.keys())

            for ref_person in person_names:
                ref_positions = employees[ref_person]

                for other_person in person_names:
                    if ref_person == other_person:
                        continue

                    other_positions = employees[other_person]

                    for common_pos in other_positions.intersection(ref_positions):
                        for excluded_pos in ref_positions - other_positions:
                            # Now: other_person has common_pos but not excluded_pos
                            question = (
                                f"Who is the {common_pos} but not {excluded_pos} "
                                f"of the company where {ref_person} is employed?"
                            )
                            answer = other_person

                            if not self._is_duplicate_question(questions, question, answer):
                                questions.append({
                                    'question': question,
                                    'answer': answer,
                                    'num_hops': 3,
                                    'num_set_operations': 2,
                                    'document_number': doc_num,
                                    'multiple_answer_dimension': 0
                                })

        return questions
    def generate_location_type_by_org_role_comparison_questions(self, graph, doc_num):
        """What is the address of [Location Type] of the company which is the [Organization Role 1] but not the [Organization Role 2] in the agreement?"""
        questions = []

        query = """
        SELECT DISTINCT ?org ?role ?loc ?loc_type
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?org <http://example.org/relation/hasLocationAt> ?loc .
            ?loc <http://example.org/isInstanceOf/> ?loc_type .
            ?org <http://example.org/isInstanceOf/> ?role .
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/")) .
            FILTER(STRSTARTS(STR(?loc_type), "http://example.org/location_type/")) .
        }
        """

        results = graph.query(query)

        # Step 1: Build data structure per organization
        org_info = {}  # org -> {roles: set(), locations: {loc_type -> set(addresses)}}

        for row in results:
            org_uri, role_uri, loc_uri, loc_type_uri = row

            org = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            location = self.clean_uri(loc_uri)
            loc_type = self.clean_uri(loc_type_uri)

            if org not in org_info:
                org_info[org] = {'roles': set(), 'locations': {}}

            org_info[org]['roles'].add(role)
            org_info[org]['locations'].setdefault(loc_type, set()).add(location)

        # Step 2: Generate comparisons across roles
        all_roles = set()
        for info in org_info.values():
            all_roles.update(info['roles'])

        all_roles = list(all_roles)

        for i in range(len(all_roles)):
            for j in range(len(all_roles)):
                if i == j:
                    continue

                role_included = all_roles[i]
                role_excluded = all_roles[j]

                matching_orgs = [
                    org for org, info in org_info.items()
                    if role_included in info['roles'] and role_excluded not in info['roles']
                ]

                # Only proceed if there's a unique match
                if len(matching_orgs) != 1:
                    continue

                org = matching_orgs[0]
                locs = org_info[org]['locations']

                for loc_type, addresses in locs.items():
                    answer = ", ".join(sorted(addresses))
                    multiple = len(addresses) > 1

                    question = (
                        f"What {'are' if multiple else 'is'} the address{'es' if multiple else ''} of "
                        f"{loc_type} of the company which is the {role_included} but not the {role_excluded} in the agreement?"
                    )

                    if not self._is_duplicate_question(questions, question, answer):
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,
                            'num_set_operations': 2,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 1 if multiple else 0
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
                
                # Generate questions
                # all_questions.extend(self.generate_multiple_position_comparison_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_position_comparison_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_exclusive_shared_role_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_exclusive_role_subrole_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_shared_role_subrole_exclusive_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_org_role_exclusion_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_person_position_questions_for_unique_dual_role_plural(graph, doc_num))
                # all_questions.extend(self.generate_dual_person_position_questions_plural(graph, doc_num)) # no answers
                # all_questions.extend(self.generate_dual_person_position_location_questions_plural(graph, doc_num)) # no answers
                # all_questions.extend(self.generate_dual_position_questions_plural(graph, doc_num)) # no answers
                # all_questions.extend(self.generate_position_by_org_role_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_position_comparison_by_org_role_questions(graph, doc_num))
                # all_questions.extend(self.generate_location_position_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_position_comparison_by_employee_questions(graph, doc_num))
                all_questions.extend(self.generate_location_type_by_org_role_comparison_questions(graph, doc_num))

            except Exception as e:
                print(f"Error processing {ttl_file}: {str(e)}")
        
        # Create DataFrame
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
        df = pd.DataFrame(all_questions, columns=columns_order)
        
        return df

def main():
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level5.csv', index=False)
    print(f"Generated {len(df)} questions and saved to level5.csv")

if __name__ == "__main__":
    main()
