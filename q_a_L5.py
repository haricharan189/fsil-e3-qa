import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from itertools import combinations

class QuestionGenerator:
    def __init__(self, ttl_dir="extracted_content_rest"):
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

    def _is_duplicate_question(self, questions, new_question, new_answer):
        """Helper function to check if a question-answer pair is an exact duplicate"""
        for q in questions:
            if new_question == q['question'] and new_answer == q['answer']:
                return True
        return False

    def generate_location_position_comparison_questions(self, graph, doc_num):
        """Generate questions comparing positions between people in companies at specific locations"""
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
                    
                    # Create question with appropriate template based on number of people
                    has_multiple = len(people_set) > 1
                    if has_multiple:
                        question = f"Who are the {common_pos}s but not {', '.join(unique_positions)} of the company associated with {location}?"
                    else:
                        question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company associated with {location}?"
                    
                    if not self._is_duplicate_question(questions, question, answer):
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  # Location -> Organization -> Person -> Position
                            'num_set_operations': 2,  # One intersection and one difference operation
                            'document_number': doc_num,
                            'multiple_answers': 1 if has_multiple else 0
                        })
        
        return questions

    def generate_position_comparison_by_employee_questions(self, graph, doc_num):
        """Generate questions about positions held by people in companies where specific employees work"""
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
                    
                    # Create question with appropriate template based on number of people
                    has_multiple = len(people_set) > 1
                    if has_multiple:
                        question = f"Who are the {common_pos}s but not {', '.join(unique_positions)} of the company where {employee_name} is employed?"
                    else:
                        question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company where {employee_name} is employed?"
                    
                    if not self._is_duplicate_question(questions, question, answer):
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  # Person -> Organization -> Other Person -> Position
                            'num_set_operations': 2,  # One intersection and one difference operation
                            'document_number': doc_num,
                            'multiple_answers': 1 if has_multiple else 0
                        })
        
        return questions

    def generate_position_by_org_role_comparison_questions(self, graph, doc_num):
        """Generate questions about positions in companies based on their organizational roles"""
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
                                has_multiple = len(people) > 1
                                if has_multiple:
                                    question = f"Who are the {position}s of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                else:
                                    question = f"Who is the {position} of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                
                                if not self._is_duplicate_question(questions, question, answer):
                                    questions.append({
                                        'question': question,
                                        'answer': answer,
                                        'num_hops': 3,  # Organization -> Role/Subrole -> Person -> Position
                                        'num_set_operations': 2,  # One intersection and one difference operation
                                        'document_number': doc_num,
                                        'multiple_answers': 1 if has_multiple else 0
                                    })
        
        return questions

    def generate_position_comparison_by_org_role_questions(self, graph, doc_num):
        """Generate questions comparing positions of people in companies with specific organizational roles"""
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
                    has_multiple = len(people_set) > 1
                    if has_multiple:
                        question = f"Who are the {common_pos}s but not {', '.join(unique_positions)} of the company which is the {role_str} in the agreement?"
                    else:
                        question = f"Who is the {common_pos} but not {', '.join(unique_positions)} of the company which is the {role_str} in the agreement?"
                    
                    if not self._is_duplicate_question(questions, question, answer):
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  # Organization -> Role/Subrole -> Person -> Position
                            'num_set_operations': 2,  # One intersection and one difference operation
                            'document_number': doc_num,
                            'multiple_answers': 1 if has_multiple else 0
                        })
        
        return questions

    def generate_location_type_by_org_role_comparison_questions(self, graph, doc_num):
        """Generate questions about location types of companies based on their organizational roles"""
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
                                
                                # Create question with appropriate template based on number of addresses
                                has_multiple = len(addresses) > 1
                                if has_multiple:
                                    question = f"What are the addresses of {loc_type} of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                else:
                                    question = f"What is the address of {loc_type} of the company which is the {common_role_str} but not the {unique_role_str} in the agreement?"
                                
                                if not self._is_duplicate_question(questions, question, answer):
                                    questions.append({
                                        'question': question,
                                        'answer': answer,
                                        'num_hops': 3,  # Organization -> Role/Subrole -> Location -> Location Type
                                        'num_set_operations': 2,  # One intersection and one difference operation
                                        'document_number': doc_num,
                                        'multiple_answers': 1 if has_multiple else 0
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
                location_position_comparison_questions = self.generate_location_position_comparison_questions(graph, doc_num)
                position_comparison_by_employee_questions = self.generate_position_comparison_by_employee_questions(graph, doc_num)
                position_by_org_role_comparison_questions = self.generate_position_by_org_role_comparison_questions(graph, doc_num)
                position_comparison_by_org_role_questions = self.generate_position_comparison_by_org_role_questions(graph, doc_num)
                location_type_by_org_role_comparison_questions = self.generate_location_type_by_org_role_comparison_questions(graph, doc_num)
                
                # Add questions to the list
                all_questions.extend(location_position_comparison_questions)
                all_questions.extend(position_comparison_by_employee_questions)
                all_questions.extend(position_by_org_role_comparison_questions)
                all_questions.extend(position_comparison_by_org_role_questions)
                all_questions.extend(location_type_by_org_role_comparison_questions)
                
            except Exception as e:
                print(f"Error processing {ttl_file}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(all_questions)
        
        # Reorder columns to put document_number first
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answers']
        df = df[columns_order]
        
        return df

def main():
    # Initialize question generator
    generator = QuestionGenerator()
    
    # Generate questions and create DataFrame
    df = generator.generate_all_questions()
    
    # Split dataframe into singular and plural questions
    df_plural = df[df['multiple_answers'] == 1]
    df_singular = df[df['multiple_answers'] == 0]
    
    # Save to CSV files
    df_singular.to_csv('qa_dataframe_L5_38.csv', index=False)
    df_plural.to_csv('qa_dataframe_L6_38.csv', index=False)
    print(f"Generated {len(df_singular)} singular questions and saved to qa_dataframe_L5.csv")
    print(f"Generated {len(df_plural)} plural questions and saved to qa_dataframe_L6.csv")
    
    # Display sample questions
    print("\nSample singular questions:")
    print(df_singular.head(2))
    print("\nSample plural questions:")
    print(df_plural.head(2))

if __name__ == "__main__":
    main()
