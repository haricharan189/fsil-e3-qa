import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from itertools import combinations

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

    def generate_position_comparison_questions(self, graph, doc_num):
        """Generate questions comparing positions between two people"""
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)
        
        # Generate all unique combinations of two people
        for person1, person2 in combinations(person_positions.keys(), 2):
            # Get positions for each person
            positions1 = person_positions[person1]
            positions2 = person_positions[person2]
            
            # Only proceed if:
            # 1. Both people have multiple positions
            # 2. They share at least one position
            common_positions = positions1.intersection(positions2)
            if len(positions1) > 1 and len(positions2) > 1 and common_positions:
                # Find positions held by person1 but not by person2
                unique_positions = positions1 - positions2
                
                # Only create question if there are unique positions
                if unique_positions:
                    # Create answer string
                    answer = ", ".join(sorted(unique_positions))
                    has_multiple = len(unique_positions) > 1
                    
                    # Create question with appropriate template based on number of positions
                    question = f"What are the positions held by {person1} but not by {person2}?" if has_multiple else f"What is the position held by {person1} but not by {person2}?"
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,  # Need to check positions of both people
                        'num_set_operations': 2,  # One set difference operation
                        'document_number': doc_num,
                        'multiple_answers': 1 if has_multiple else 0
                    })
        
        return questions
    
    def generate_multi_position_questions(self, graph, doc_num):
        """Generate questions about people holding multiple positions in organizations"""
        questions = []
        
        # Get all organizations and their people with positions
        org_people_positions = self.get_org_people_positions(graph)
        
        # For each organization
        for org_name, people in org_people_positions.items():
            # Find people with multiple positions
            for person_name, positions in people.items():
                if len(positions) >= 2:
                    # Generate combinations of positions
                    for pos1, pos2 in combinations(sorted(positions), 2):
                        # Get all people who hold both positions
                        holders = set()
                        for p, pos in people.items():
                            if pos1 in pos and pos2 in pos:
                                holders.add(p)
                        
                        # Create answer string
                        answer = ", ".join(sorted(holders))
                        
                        # Create question with appropriate template based on number of holders
                        has_multiple = len(holders) > 1
                        question = f"Who are the {pos1}s and {pos2}s of {org_name}?" if has_multiple else f"Who is the {pos1} and {pos2} of {org_name}?"
                        
                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 2,  # Need to check both positions
                            'num_set_operations': 1,  # Need to find intersection of people with both positions
                            'document_number': doc_num,
                            'multiple_answers': 1 if has_multiple else 0
                        })
        
        return questions
    
    def generate_location_position_questions(self, graph, doc_num):
        """Generate questions about people's positions in companies at specific locations"""
        questions = {}  # Using dict to group answers by question
        
        # Get all locations, organizations and their people with positions
        location_org_people = self.get_location_org_people_positions(graph)
        
        # For each location
        for location, orgs in location_org_people.items():
            # Track all positions and their holders at this location
            position_holders = {}  # position -> set of people
            
            # For each organization at this location
            for org_name, people in orgs.items():
                # For each person in the organization
                for person_name, positions in people.items():
                    # Add person to each position they hold
                    for position in positions:
                        # Create question with appropriate template based on number of holders
                        base_question = f"Who is the {position} of the company associated with {location}?"
                        
                        if base_question not in position_holders:
                            position_holders[base_question] = set()
                        position_holders[base_question].add(person_name)
            
            # Create questions with combined answers
            for base_question, people in position_holders.items():
                # Sort names alphabetically and join with commas
                answer = ", ".join(sorted(people))
                
                # Create question with appropriate template based on number of holders
                has_multiple = len(people) > 1
                if has_multiple:
                    # Extract position from base question
                    position_start = base_question.find("the ") + 4
                    position_end = base_question.find(" of")
                    position = base_question[position_start:position_end]
                    # Add 's' to position and replace "Who is the" with "Who are the"
                    pluralized = base_question.replace(f"Who is the {position}", f"Who are the {position}s")
                    question = pluralized
                else:
                    question = base_question
                
                questions[question] = {
                    'question': question,
                    'answer': answer,
                    'num_hops': 3,  # Location -> Organization -> Person -> Position
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answers': 1 if has_multiple else 0
                }
        
        return list(questions.values())
    
    def generate_employee_org_position_questions(self, graph, doc_num):
        """Generate questions about positions in companies where specific people work"""
        questions = {}  # Using dict to group answers by question
        
        # Get all organizations with their employees and positions
        org_info = self.get_org_employees_positions(graph)
        
        # For each organization
        for org_name, info in org_info.items():
            # For each employee in the organization
            for employee in info['employees']:
                # Get this employee's positions
                employee_positions = set()
                for pos, holders in info['positions'].items():
                    if employee in holders:
                        employee_positions.add(pos)
                
                # For each position in the organization
                for position, holders in info['positions'].items():
                    # Skip if this is a position held by the employee we're asking about
                    if position in employee_positions:
                        continue
                        
                    # Create question with appropriate template based on number of holders
                    base_question = f"Who is the {position} of the company where {employee} is employed?"
                    
                    # Sort and join position holders
                    answer = ", ".join(sorted(holders))
                    
                    # Create question with appropriate template based on number of holders
                    has_multiple = len(holders) > 1
                    if has_multiple:
                        # Extract position from base question
                        position_start = base_question.find("the ") + 4
                        position_end = base_question.find(" of")
                        position = base_question[position_start:position_end]
                        # Add 's' to position and replace "Who is the" with "Who are the"
                        pluralized = base_question.replace(f"Who is the {position}", f"Who are the {position}s")
                        question = pluralized
                    else:
                        question = base_question
                    
                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 3,  # Person -> Organization -> Other Person -> Position
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answers': 1 if has_multiple else 0
                    }
        
        return list(questions.values())
    

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

    def generate_org_role_comparison_questions(self, graph, doc_num):
        """Generate questions comparing roles between organizations"""
        questions = []
        
        # Get all organizations with their roles and subroles
        org_roles = self.get_org_roles_and_subroles(graph)
        
        # Generate all unique combinations of two organizations
        for org1, org2 in combinations(org_roles.keys(), 2):
            # Get roles for each organization
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]
            
            # Only proceed if both organizations have roles
            if not roles1 or not roles2:
                continue
                
            # Find common roles between organizations
            common_roles = set(roles1.keys()) & set(roles2.keys())
            
            # Only proceed if:
            # 1. Both organizations have multiple roles (>1)
            # 2. They share at least one role
            if common_roles and len(roles1) > 1 and len(roles2) > 1:
                # Find roles unique to org1
                unique_roles = set(roles1.keys()) - set(roles2.keys())
                
                # Store all unique role/subrole combinations
                all_unique_combinations = []
                
                # Add unique roles without subroles
                for role in unique_roles:
                    subroles = roles1[role]
                    if subroles:
                        # Add each subrole combination
                        for subrole in subroles:
                            all_unique_combinations.append(f"{subrole} {role}")
                    else:
                        # Add just the role
                        all_unique_combinations.append(role)
                
                # For common roles, check for different subroles
                for role in common_roles:
                    subroles1 = roles1[role]
                    subroles2 = roles2[role]
                    
                    # Find subroles unique to org1
                    unique_subroles = subroles1 - subroles2
                    
                    # Add unique subrole combinations
                    for subrole in unique_subroles:
                        all_unique_combinations.append(f"{subrole} {role}")
                
                # Only create question if there are unique combinations
                if all_unique_combinations:
                    # Sort combinations for consistent output
                    all_unique_combinations.sort()
                    
                    # Create answer string
                    answer = ", ".join(all_unique_combinations)
                    
                    # Create question with appropriate template based on number of answers
                    has_multiple = len(all_unique_combinations) > 1
                    question = f"What role does {org1} have in the agreement which is not the role of {org2}?" if not has_multiple else f"What roles does {org1} have in the agreement which are not the roles of {org2}?"
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,  # Organization -> Role/Subrole
                        'num_set_operations': 1,  # One set difference operation
                        'document_number': doc_num,
                        'multiple_answers': 1 if has_multiple else 0
                    })
        
        return questions

    def generate_person_position_by_org_role_questions(self, graph, doc_num):
        """Generate questions about people's positions in companies with specific organizational roles"""
        questions = []
        
        # Get role-subrole mapping
        role_subrole_map = self.get_org_role_subrole_map(graph)
        
        # SPARQL query to get organizations with their roles/sub-roles and their employees with positions
        query = """
        SELECT DISTINCT ?role ?sub_role ?org ?person ?position
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org ;
                   <http://example.org/isInstanceOf/> ?position .
            {
                ?org <http://example.org/isInstanceOf/> ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
                OPTIONAL {
                    ?org <http://example.org/isInstanceOf/> ?sub_role .
                    ?sub_role rdfs:subClassOf ?role .
                    FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
                }
            }
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        """
        
        results = graph.query(query)
        
        # Create a mapping to store all the data
        role_person_map = {}  # (role, sub_role, position) -> set of person names
        
        for row in results:
            role_uri = row[0]
            sub_role_uri = row[1] if row[1] else None
            person_uri = row[3]
            position_uri = row[4]
            
            # Clean the URIs to get readable text
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri)
            
            # Create key for role combination and position
            role_key = (role, sub_role, position)
            
            # Store person for this role combination and position
            if role_key not in role_person_map:
                role_person_map[role_key] = set()
            role_person_map[role_key].add(person_name)
        
        # Generate questions for each role combination and position
        for (role, sub_role, position), persons in role_person_map.items():
            # Skip if no persons found
            if not persons:
                continue
                
            # Sort and join person names
            answer = ", ".join(sorted(persons))
            
            # Create question with appropriate template based on number of persons
            has_multiple = len(persons) > 1
            if sub_role:  # Has both sub-role and role
                question = f"Who is the {position} of the company which is the {sub_role} {role} in the agreement?"
                if has_multiple:
                    question = f"Who are the {position}s of the company which is the {sub_role} {role} in the agreement?"
            else:  # Only has main role
                question = f"Who is the {position} of the company which is the {role} in the agreement?"
                if has_multiple:
                    question = f"Who are the {position}s of the company which is the {role} in the agreement?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 3,  # Role -> Organization -> Person -> Position
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

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

    def generate_org_role_subrole_comparison_questions(self, graph, doc_num):
        """Generate questions comparing organizations based on their roles and subroles"""
        questions = []
        
        # Get role-organization mapping
        role_org_map = self.get_role_org_mapping(graph)
        
        # Generate combinations of role pairs
        role_pairs = list(combinations(role_org_map.keys(), 2))
        
        for role_pair1, role_pair2 in role_pairs:
            # Get organizations for each role combination
            orgs1 = role_org_map[role_pair1]
            orgs2 = role_org_map[role_pair2]
            
            # Check if both have multiple organizations and share at least one
            if len(orgs1) > 1 and len(orgs2) > 1 and orgs1.intersection(orgs2):
                # Find organizations unique to first role combination
                unique_orgs = orgs1 - orgs2
                
                if unique_orgs:  # Only create question if there are unique organizations
                    # Format role strings
                    role1, sub_role1 = role_pair1
                    role2, sub_role2 = role_pair2
                    
                    role1_str = f"{sub_role1} {role1}" if sub_role1 else role1
                    role2_str = f"{sub_role2} {role2}" if sub_role2 else role2
                    
                    # Create answer string
                    answer = ", ".join(sorted(unique_orgs))
                    
                    # Create question with appropriate template based on number of answers
                    has_multiple = len(unique_orgs) > 1
                    if has_multiple:
                        question = f"What companies are the {role1_str} but not the {role2_str} in the agreement?"
                    else:
                        question = f"What company is the {role1_str} but not the {role2_str} in the agreement?"
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,  # Organization -> Role/Subrole
                        'num_set_operations': 1,  # One set difference operation
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
                
                # Generate all types of questions
                position_diff_questions = self.generate_position_comparison_questions(graph, doc_num)
                multi_position_questions = self.generate_multi_position_questions(graph, doc_num)
                location_position_questions = self.generate_location_position_questions(graph, doc_num)
                employee_org_position_questions = self.generate_employee_org_position_questions(graph, doc_num)
                person_position_by_org_role_questions = self.generate_person_position_by_org_role_questions(graph, doc_num)
                org_role_comparison_questions = self.generate_org_role_comparison_questions(graph, doc_num)
                org_role_subrole_comparison_questions = self.generate_org_role_subrole_comparison_questions(graph, doc_num)
                
                # # Combine all questions
                all_questions.extend(position_diff_questions)
                all_questions.extend(multi_position_questions)
                all_questions.extend(location_position_questions)
                all_questions.extend(employee_org_position_questions)
                all_questions.extend(person_position_by_org_role_questions)
                all_questions.extend(org_role_comparison_questions)
                all_questions.extend(org_role_subrole_comparison_questions)
                
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
    
    # Save to CSV
    df.to_csv('qa_dataframe_L3.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe_L3.csv")
    
    # Display sample questions of each type
    print("\nSample position comparison questions:")
    print(df[df['question'].str.startswith('What is the position')].head(2))
    print("\nSample multi-position questions:")
    print(df[df['question'].str.startswith('Who is the') & ~df['question'].str.contains('company associated')].head(2))
    print("\nSample location-position questions:")
    print(df[df['question'].str.contains('company associated with')].head(2))
    print("\nSample employee-org-position questions:")
    print(df[df['question'].str.contains('company associated where')].head(2))

if __name__ == "__main__":
    main()
