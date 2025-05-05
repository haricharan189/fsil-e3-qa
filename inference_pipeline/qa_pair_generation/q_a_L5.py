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
        """[Level 5] What roles do [Org Name 1] and [Org Name 2] have in the agreement which is not the role of [Org Name 3]?"""

        questions = []
        seen_questions = set()

        org_roles = self.get_org_roles_with_subroles(graph)
        valid_orgs = {org: roles for org, roles in org_roles.items() if len(roles) > 1}

        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]

            # Ensure Org A shares at least one role-subrole with Org B
            shared_12 = roles1 & roles2
            if not shared_12:
                continue  

            # Ensure Org A shares at least one role-subrole with Org C
            shared_13 = roles1 & roles3
            if not shared_13:
                continue  

            # Find role-subroles shared by Org1 & Org2 but NOT by Org3
            exclusive_shared_roles = shared_12 - roles3

            # Filter out pairs where subrole is None
            exclusive_shared_roles = {(role, subrole) for role, subrole in exclusive_shared_roles if subrole is not None}

            if len(exclusive_shared_roles) > 1:
                role_subrole_str = ", ".join([f"{subrole} {role}" for role, subrole in exclusive_shared_roles])
                question = f"What are the roles and subroles that {org1} and {org2} share in the agreement that are not held by {org3}?"

                key = (frozenset([org1, org2]), frozenset(exclusive_shared_roles), org3)
                if key in seen_questions:
                    continue  
                seen_questions.add(key)

                print(f"Plural: {question} | Answer: {role_subrole_str}")

                questions.append({
                    'question': question,
                    'answer': role_subrole_str,
                    'num_hops': 1,
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1
                })

        return questions
    
    def generate_exclusive_role_subrole_questions_plural(self, graph, doc_num):
        """
        What are the roles and subroles that [Org Name 1] has in the agreement which are not held by [Org Name 2] or [Org Name 3]? [if more than one]
        """
        questions = []
        seen_questions = set()

        org_roles = self.get_org_roles_with_subroles(graph)

        # Clean roles: only include if subrole exists and isn't just "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.strip().lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Filter: only orgs with more than one such (role, subrole)
        orgs_with_multiple_roles = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) > 1}

        for org1, org2, org3 in combinations(orgs_with_multiple_roles.keys(), 3):
            roles1 = orgs_with_multiple_roles[org1]
            roles2 = orgs_with_multiple_roles[org2]
            roles3 = orgs_with_multiple_roles[org3]

            # Org1 must share at least one role-subrole with both Org2 and Org3
            if not (roles1 & roles2 and roles1 & roles3):
                continue

            exclusive_roles = roles1 - (roles2 | roles3)

            # Only continue if more than one exclusive role-subrole exists
            if len(exclusive_roles) <= 1:
                continue

            # Clean into readable strings
            sorted_exclusive = sorted(f"{sub} {role}".strip() for role, sub in exclusive_roles)
            answer = ", ".join(sorted_exclusive[:-1]) + f", and {sorted_exclusive[-1]}" if len(sorted_exclusive) > 2 else " and ".join(sorted_exclusive)

            question = (
                f"What are the roles and subroles that {org1} holds in the agreement that are not held by {org2} or {org3}?"
            )

            key = (org1, frozenset(exclusive_roles), frozenset([org2, org3]))
            if key in seen_questions:
                continue
            seen_questions.add(key)

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
        [Level 5] What companies are the [Subrole1 Role1] and [Subrole2 Role2] 
        but not the [Subrole3 Role3] in the agreement?
        """
        questions = []

        # 1) Build (role, subrole) → set of orgs
        org_role_pairs = self.get_org_roles_with_subroles(graph)
        pair_to_orgs = defaultdict(set)
        for org, roles in org_role_pairs.items():
            for role, sub in roles:
                pair_to_orgs[(role, sub)].add(org)

        # 2) Triple‑loop with meaningful‑“but not” guard
        for (r1, s1), (r2, s2), (r3, s3) in combinations(pair_to_orgs, 3):
            shared   = pair_to_orgs[(r1, s1)] & pair_to_orgs[(r2, s2)]
            excluded = pair_to_orgs[(r3, s3)]

            # only keep if:
            #  - some orgs have A∧B,
            #  - at least one of those also has C (so "but not C" removes something),
            #  - and at least one remains after removal
            if shared and (shared & excluded) and (shared - excluded):
                result = shared - excluded
                if len(result) > 1:  # plural case
                    companies = sorted(result)
                    answer = ", ".join(companies)

                    fmt = lambda role, sub: f"{sub} {role}".strip() if sub else role
                    q_text = (
                        f"What companies are the {fmt(r1, s1)} and the {fmt(r2, s2)} "
                        f"but not the {fmt(r3, s3)} in the agreement?"
                    )

                    questions.append({
                        'document_number':           doc_num,
                        'question':                  q_text,
                        'answer':                    answer,
                        'num_hops':                  1,
                        'num_set_operations':        3,
                        'multiple_answer_dimension': 1
                    })

        return questions


    
    def generate_org_role_exclusion_questions_plural(self, graph, doc_num):
        """
        What companies are the [Role/Subrole1] but not the [Role/Subrole2] or [Role/Subrole3] in the agreement?
        """
        questions = []
        org_roles = self.get_org_roles_with_subroles(graph)

        # (role, subrole) → set(orgs)
        pair_to_orgs = defaultdict(set)
        for org, pairs in org_roles.items():
            for role, sub in pairs:
                # Skip pairs with no subrole
                if not sub or sub.strip().lower() == "none":
                    continue
                pair_to_orgs[(role, sub)].add(org)

        def fmt(role, sub):
            return f"{sub} {role}".strip()

        for (r1, s1), (r2, s2), (r3, s3) in combinations(pair_to_orgs.keys(), 3):
            include_orgs = pair_to_orgs[(r1, s1)]
            exclude_orgs = pair_to_orgs[(r2, s2)] | pair_to_orgs[(r3, s3)]

            # Make sure there's at least one org that overlaps between include and exclude (shared context)
            if not (include_orgs & exclude_orgs):
                continue

            exclusive = include_orgs - exclude_orgs

            # Only ask plural questions when multiple companies satisfy the condition
            if len(exclusive) > 1:
                answer = ", ".join(sorted(exclusive))
                q_text = (
                    f"What companies are the {fmt(r1, s1)} "
                    f"but not the {fmt(r2, s2)} or {fmt(r3, s3)} in the agreement?"
                )
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
        """Who are the [Position]s of the company(ies) which are both the [Org Role(+Subrole) 1] and the [Org Role(+Subrole) 2] in the agreement?"""
        questions = []
        seen = set()

        # 1) Fetch org → employees+positions
        org_info  = self.get_org_employees_positions(graph)
        # 2) Fetch org → (role, subrole) sets
        raw_roles = self.get_org_roles_with_subroles(graph)

        print(f"DEBUG: Retrieved {len(raw_roles)} orgs with roles from graph.")

        # 3) Clean out empty/None subroles
        cleaned_roles = {}
        for org, pairs in raw_roles.items():
            filtered = {(r, s) for (r, s) in pairs if s and s.strip().lower() != "none"}
            if filtered:
                cleaned_roles[org] = filtered
        print(f"DEBUG: After cleaning, {len(cleaned_roles)} orgs have non-empty subroles.")

        # 4) Build mapping: frozenset({(r1,s1),(r2,s2)}) → set of orgs
        pair_to_orgs = defaultdict(set)
        for org, roles in cleaned_roles.items():
            for combo in combinations(roles, 2):
                key = frozenset(combo)
                pair_to_orgs[key].add(org)
        print(f"DEBUG: Generated {len(pair_to_orgs)} unique role-pair combos.")

        # 5) For each role-pair combo, process all matching orgs
        for role_pair, orgs in pair_to_orgs.items():
            (r1, s1), (r2, s2) = tuple(role_pair)
            print(f"DEBUG: Role pair {role_pair} matches orgs: {orgs}")

            for the_org in orgs:
                if the_org not in org_info:
                    print(f"DEBUG: Skipping {the_org}—no position data.")
                    continue

                # 6) For each position with multiple holders
                for position, holders in org_info[the_org]['positions'].items():
                    print(f"DEBUG: Org {the_org}, position {position} has holders: {holders}")
                    if len(holders) < 2:
                        print(f"DEBUG: Skipping position {position}—needs ≥2 holders.")
                        continue

                    fmt = lambda r, s: f"{s} {r}".strip() if s else r
                    q_text = (
                        f"Who are the {position}s of the company which is both the "
                        f"{fmt(r1, s1)} and the {fmt(r2, s2)} in the agreement?"
                    )
                    answer = ", ".join(sorted(holders))

                    key = (q_text, answer)
                    if key in seen:
                        print(f"DEBUG: Duplicate Q&A, skipping.")
                        continue
                    seen.add(key)

                    print(f"Generated question: {q_text} | Answer: {answer}")

                    questions.append({
                        'document_number':           doc_num,
                        'question':                  q_text,
                        'answer':                    answer,
                        'num_hops':                  3,
                        'num_set_operations':        1,
                        'multiple_answer_dimension': 1
                    })

        print(f"DEBUG: Total generated questions: {len(questions)}")
        return questions
    
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
        Who is the [Person Position] of the company which is the [Org Role(-s) (+ Sub-Role(-s)) 1] but not the [Org Role(-s) (+ Sub-Role(-s)) 2] in the agreement? [if one, and the company should be uniquely identifiable]
        """
        questions = []
        
        # First get all organizations with their roles and subroles
        org_roles_query = """
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
        
        results = graph.query(org_roles_query)
        
        # Create mappings for organizations and their roles/subroles/people
        org_info = {}  # org -> {roles: set(), role_subroles: set(), positions: {position: set(people)}}
        
        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            sub_role_uri = row[2] if row[2] else None
            person_uri = row[3]
            position_uri = row[4]
            
            # Clean the URIs
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Initialize organization entry if not exists
            if org_name not in org_info:
                org_info[org_name] = {
                    'roles': set(),  # Just roles
                    'role_subroles': set(),  # (role, sub_role) pairs
                    'positions': {},  # position -> set of people
                    'roles_with_subroles': set()  # roles that have subroles
                }
            
            # Add role information
            org_info[org_name]['roles'].add(role)
            if sub_role:
                org_info[org_name]['role_subroles'].add((role, sub_role))
                org_info[org_name]['roles_with_subroles'].add(role)
            
            # Add person to position
            if position not in org_info[org_name]['positions']:
                org_info[org_name]['positions'][position] = set()
            org_info[org_name]['positions'][position].add(person_name)
        
        # Generate questions by comparing organizations
        orgs = list(org_info.keys())
        for i, org1 in enumerate(orgs):
            for org2 in orgs[i+1:]:
                org1_info = org_info[org1]
                org2_info = org_info[org2]
                
                # Create role combinations, excluding role-only entries when subroles exist
                all_role_combinations1 = set()
                all_role_combinations2 = set()
                
                # Add role-subrole pairs
                all_role_combinations1.update(org1_info['role_subroles'])
                all_role_combinations2.update(org2_info['role_subroles'])
                
                # Add roles without subroles
                for role in org1_info['roles']:
                    if role not in org1_info['roles_with_subroles']:
                        all_role_combinations1.add((role, None))
                        
                for role in org2_info['roles']:
                    if role not in org2_info['roles_with_subroles']:
                        all_role_combinations2.add((role, None))
                
                # Find common role combinations
                common_combinations = all_role_combinations1.intersection(all_role_combinations2)
                
                if common_combinations:  # Only proceed if they share at least one role/role-subrole
                    # For each common combination
                    for common_combo in common_combinations:
                        common_role, common_subrole = common_combo
                        
                        # Find unique role combinations for org1
                        unique_combinations1 = all_role_combinations1 - all_role_combinations2
                        
                        # For each unique combination in org1
                        for unique_combo in unique_combinations1:
                            unique_role, unique_subrole = unique_combo
                            
                            # For each position in org1
                            for position, people in org1_info['positions'].items():
                                # Create role description strings
                                if common_subrole:
                                    common_role_str = f"{common_subrole} {common_role}"
                                else:
                                    common_role_str = common_role
                                    
                                if unique_subrole:
                                    unique_role_str = f"{unique_subrole} {unique_role}"
                                else:
                                    unique_role_str = unique_role
                                
                                # Create answer string
                                answer = ", ".join(sorted(people))
                                
                                # Create question with appropriate template based on number of people
                
                                question = f"Who is the {position} of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                
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

        """Who is the [Person Position 1] but not [Person Position 2] of the company which is the [Org Role(-s) (+ Sub-Role(-s))] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = []
        
        # Get organizations with their roles, subroles, and people's positions
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
        
        # Create mappings for organizations and their roles/subroles/people
        org_info = {}  # org -> {roles: set(), role_subroles: set(), people: {person: set(positions)}, roles_with_subroles: set()}
        
        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            sub_role_uri = row[2] if row[2] else None
            person_uri = row[3]
            position_uri = row[4]
            
            # Clean the URIs
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Initialize organization entry if not exists
            if org_name not in org_info:
                org_info[org_name] = {
                    'roles': set(),  # Just roles
                    'role_subroles': set(),  # (role, sub_role) pairs
                    'people': {},  # person -> set of positions
                    'roles_with_subroles': set()  # roles that have subroles
                }
            
            # Add role information
            org_info[org_name]['roles'].add(role)
            if sub_role:
                org_info[org_name]['role_subroles'].add((role, sub_role))
                org_info[org_name]['roles_with_subroles'].add(role)
            
            # Add person and position
            if person_name not in org_info[org_name]['people']:
                org_info[org_name]['people'][person_name] = set()
            org_info[org_name]['people'][person_name].add(position)
        
        # For each organization
        for org_name, info in org_info.items():
            # Get role descriptions (prioritizing subrole-role pairs)
            role_descriptions = []
            
            # First add role-subrole pairs
            for role, sub_role in info['role_subroles']:
                role_descriptions.append(f"{sub_role} {role}")
            
            # Then add roles without subroles
            for role in info['roles']:
                if role not in info['roles_with_subroles']:
                    role_descriptions.append(role)
            
            # For each role description
            for role_str in role_descriptions:
                # For each pair of people in the organization
                people = info['people']
                person_names = list(people.keys())
                
                # Track position combinations and the people who have them
                position_combo_people = {}  # (common_pos, unique_positions_tuple) -> set of people
                
                # Compare each pair of people
                for i, person1 in enumerate(person_names):
                    positions1 = people[person1]
                    
                    # Only consider people with multiple positions
                    if len(positions1) <= 1:
                        continue
                        
                    for person2 in person_names[i+1:]:
                        positions2 = people[person2]
                        
                        # Only consider people with multiple positions
                        if len(positions2) <= 1:
                            continue
                        
                        # Find common positions
                        common_positions = positions1.intersection(positions2)
                        
                        # Only proceed if they share at least one position
                        if common_positions:
                            # For each common position
                            for common_pos in common_positions:
                                # Find positions unique to person1
                                unique_positions1 = positions1 - positions2
                                if unique_positions1:
                                    # Create key for this position combination
                                    unique_pos_tuple = tuple(sorted(unique_positions1))
                                    pos_key = (common_pos, unique_pos_tuple)
                                    
                                    # Add person1 to the set of people for this position combination
                                    if pos_key not in position_combo_people:
                                        position_combo_people[pos_key] = set()
                                    position_combo_people[pos_key].add(person1)
                                
                                # Find positions unique to person2
                                unique_positions2 = positions2 - positions1
                                if unique_positions2:
                                    # Create key for this position combination
                                    unique_pos_tuple = tuple(sorted(unique_positions2))
                                    pos_key = (common_pos, unique_pos_tuple)
                                    
                                    # Add person2 to the set of people for this position combination
                                    if pos_key not in position_combo_people:
                                        position_combo_people[pos_key] = set()
                                    position_combo_people[pos_key].add(person2)
                
                # Generate questions for each position combination
                for (common_pos, unique_positions), people_set in position_combo_people.items():
                    # Create answer string
                    answer = ", ".join(sorted(people_set))
                    
                    # Create question with appropriate template based on number of people
                    question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company which is the {role_str} in the agreement?"
                    
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
        Who is the [Person Position 1] but not [Person Position 2] of the company associated with [Location]? [if one]
        """
        questions = []
        
        # Get all locations, organizations and their people with positions
        location_org_people = self.get_location_org_people_positions(graph)
        
        # For each location
        for location, orgs in location_org_people.items():
            # For each organization at this location
            for org_name, people in orgs.items():
                # Track people for each position combination
                position_combo_people = {}  # (common_pos, unique_pos_tuple) -> set of people
                
                # Generate all unique combinations of two people
                for person1, person2 in combinations(people.keys(), 2):
                    # Get positions for each person
                    positions1 = people[person1]
                    positions2 = people[person2]
                    
                    # Only proceed if:
                    # 1. Both people have multiple positions
                    # 2. They share at least one position
                    common_positions = positions1.intersection(positions2)
                    if len(positions1) > 1 and len(positions2) > 1 and common_positions:
                        # For each common position
                        for common_pos in common_positions:
                            # Find positions held by person1 but not by person2
                            unique_positions = positions1 - positions2
                            
                            # Only proceed if there are unique positions
                            if unique_positions:
                                # Create key for this position combination
                                unique_pos_tuple = tuple(sorted(unique_positions))
                                pos_key = (common_pos, unique_pos_tuple)
                                
                                # Add person1 to the set of people for this position combination
                                if pos_key not in position_combo_people:
                                    position_combo_people[pos_key] = set()
                                position_combo_people[pos_key].add(person1)
                
                # Generate questions for each position combination
                for (common_pos, unique_positions), people_set in position_combo_people.items():
                    # Create answer string
                    answer = ", ".join(sorted(people_set))
                    question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company associated with {location}?"
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
        """Who is the [Person Position 1] but not [Person Position 2] of the company associated where [Person Name] is employed? [if one]"""
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
        
        # Create a mapping of organizations to their employees and positions
        org_employee_positions = {}
        
        for row in results:
            org_uri = row[0]
            person_uri = row[1]
            position_uri = row[2]
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Initialize organization entry if not exists
            if org_name not in org_employee_positions:
                org_employee_positions[org_name] = {}
            
            # Initialize person entry if not exists
            if person_name not in org_employee_positions[org_name]:
                org_employee_positions[org_name][person_name] = set()
            
            # Add position
            org_employee_positions[org_name][person_name].add(position)
        
        # For each organization
        for org_name, employees in org_employee_positions.items():
            # For each employee (who will be referenced in the question)
            for employee_name, employee_positions in employees.items():
                # Track position combinations and the people who have them
                position_combo_people = {}  # (common_pos, unique_positions_tuple) -> set of people
                
                # For each other employee in the same organization
                for other_person, other_positions in employees.items():
                    # Skip if it's the same person
                    if other_person == employee_name:
                        continue
                    
                    # Check if they share at least one position
                    common_positions = employee_positions.intersection(other_positions)
                    
                    # Only proceed if:
                    # 1. Both people have multiple positions
                    # 2. They share at least one position
                    if len(other_positions) > 1 and common_positions:
                        # For each common position
                        for common_pos in common_positions:
                            # Find positions held by other_person but not by employee_name
                            unique_positions = other_positions - employee_positions
                            
                            # Only proceed if there are unique positions
                            if unique_positions:
                                # Create key for this position combination
                                unique_pos_tuple = tuple(sorted(unique_positions))
                                pos_key = (common_pos, unique_pos_tuple)
                                
                                # Add other_person to the set of people for this position combination
                                if pos_key not in position_combo_people:
                                    position_combo_people[pos_key] = set()
                                position_combo_people[pos_key].add(other_person)
                
                # Generate questions for each position combination
                for (common_pos, unique_positions), people_set in position_combo_people.items():
                    # Skip if the employee_name is in the answer set
                    if employee_name in people_set:
                        continue
                        
                    # Create answer string
                    answer = ", ".join(sorted(people_set))
                    question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company where {employee_name} is employed?"
                    
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
        """What is the [Location Type] office of the company which is both the [Org Role(-s) (+ Sub-Role(-s)) 1] but not the [Org Role(-s) (+ Sub-Role(-s)) 2] in the agreement? [if one, and the company should be uniquely identifiable]"""
        questions = []
        
        # Get organizations with their roles, subroles, and locations with types
        query = """
        SELECT DISTINCT ?org ?role ?sub_role ?loc ?loc_type
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?org <http://example.org/relation/hasLocationAt> ?loc .
            ?loc <http://example.org/isInstanceOf/> ?loc_type .
            {
                ?org <http://example.org/isInstanceOf/> ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            }
            OPTIONAL {
                ?org <http://example.org/isInstanceOf/> ?sub_role .
                ?sub_role rdfs:subClassOf ?role .
                FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
            }
            FILTER(STRSTARTS(STR(?loc_type), "http://example.org/location_type/"))
        }
        """
        
        results = graph.query(query)
        
        # Create mappings for organizations and their roles/subroles/locations
        org_info = {}  # org -> {roles: set(), role_subroles: set(), locations: {loc_type: set(addresses)}, roles_with_subroles: set()}
        
        for row in results:
            org_uri = row[0]
            role_uri = row[1]
            sub_role_uri = row[2] if row[2] else None
            loc_uri = row[3]
            loc_type_uri = row[4]
            
            # Clean the URIs
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            location = self.clean_uri(loc_uri)
            loc_type = self.clean_uri(loc_type_uri)
            
            # Initialize organization entry if not exists
            if org_name not in org_info:
                org_info[org_name] = {
                    'roles': set(),  # Just roles
                    'role_subroles': set(),  # (role, sub_role) pairs
                    'locations': {},  # loc_type -> set of addresses
                    'roles_with_subroles': set()  # roles that have subroles
                }
            
            # Add role information
            org_info[org_name]['roles'].add(role)
            if sub_role:
                org_info[org_name]['role_subroles'].add((role, sub_role))
                org_info[org_name]['roles_with_subroles'].add(role)
            
            # Add location
            if loc_type not in org_info[org_name]['locations']:
                org_info[org_name]['locations'][loc_type] = set()
            org_info[org_name]['locations'][loc_type].add(location)
        
        # Generate questions by comparing organizations
        orgs = list(org_info.keys())
        for i, org1 in enumerate(orgs):
            for org2 in orgs[i+1:]:
                org1_info = org_info[org1]
                org2_info = org_info[org2]
                
                # Create role combinations, excluding role-only entries when subroles exist
                all_role_combinations1 = set()
                all_role_combinations2 = set()
                
                # Add role-subrole pairs first
                all_role_combinations1.update(org1_info['role_subroles'])
                all_role_combinations2.update(org2_info['role_subroles'])
                
                # Add roles without subroles
                for role in org1_info['roles']:
                    if role not in org1_info['roles_with_subroles']:
                        all_role_combinations1.add((role, None))
                        
                for role in org2_info['roles']:
                    if role not in org2_info['roles_with_subroles']:
                        all_role_combinations2.add((role, None))
                
                # Find common role combinations
                common_combinations = all_role_combinations1.intersection(all_role_combinations2)
                
                if common_combinations:  # Only proceed if they share at least one role/role-subrole
                    # For each common combination
                    for common_combo in common_combinations:
                        common_role, common_subrole = common_combo
                        
                        # Find unique role combinations for org1
                        unique_combinations1 = all_role_combinations1 - all_role_combinations2
                        
                        # For each unique combination in org1
                        for unique_combo in unique_combinations1:
                            unique_role, unique_subrole = unique_combo
                            
                            # For each location type and its addresses in org1
                            for loc_type, addresses in org1_info['locations'].items():
                                # Create role description strings
                                if common_subrole:
                                    common_role_str = f"{common_subrole} {common_role}"
                                else:
                                    common_role_str = common_role
                                    
                                if unique_subrole:
                                    unique_role_str = f"{unique_subrole} {unique_role}"
                                else:
                                    unique_role_str = unique_role
                                
                                # Create answer string
                                answer = ", ".join(sorted(addresses))
                                question = f"What is the address of {loc_type} of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                
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
                print("debug")
                all_questions.extend(self.generate_multiple_position_comparison_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_position_comparison_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_exclusive_shared_role_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_exclusive_role_subrole_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_shared_role_subrole_exclusive_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_org_role_exclusion_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_person_position_questions_for_unique_dual_role_plural(graph, doc_num))
                all_questions.extend(self.generate_dual_person_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_dual_person_position_location_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_dual_position_questions_plural(graph, doc_num))
                all_questions.extend(self.generate_position_by_org_role_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_position_comparison_by_org_role_questions(graph, doc_num))
                all_questions.extend(self.generate_location_position_comparison_questions(graph, doc_num))
                all_questions.extend(self.generate_position_comparison_by_employee_questions(graph, doc_num))
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
