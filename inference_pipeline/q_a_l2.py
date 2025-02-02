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


    def generate_shared_position_questions(self, graph, doc_num):
        """Generate questions about a single person's positions"""
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)
        
        for person, positions in person_positions.items():
            if len(positions) == 1:  # Ensure only one position is extracted
                answer = ", ".join(positions)  # Convert set to string
            
                question = f"What position does {person} hold?"
                
                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1,  
                    'num_set_operations': 0,  
                    'document_number': doc_num
                })
        
        return questions
    
    def generate_shared_position_questions_plural(self, graph, doc_num):
        """Generate questions about shared positions between two people"""
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)
        
        # Proceed only if there are at least two people
        if len(person_positions) > 1:
            # Generate all unique combinations of two people
            for person1, person2 in combinations(person_positions.keys(), 2):
                positions1 = person_positions[person1]
                positions2 = person_positions[person2]
                
                # Find positions held by both persons
                shared_positions = positions1 & positions2
                
                if len(shared_positions)>1:
                    answer = ", ".join(sorted(shared_positions))  # Convert set to string
                    
                    # Create question
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
            
    from itertools import combinations

    def generate_shared_role_questions(self, graph, doc_num):
        """Generate questions about shared roles between two organizations in an agreement"""
        questions = []
        
        # Get all organizations and their roles
        org_roles = self.get_org_roles(graph)
        
        # Generate all unique combinations of two organizations
        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]
            
            # Ensure each organization has more than one role
            if len(roles1) > 1 and len(roles2) > 1:
                # Find roles shared by both organizations
                shared_roles = roles1 & roles2
                
                # Only proceed if exactly ONE role is shared
                if len(shared_roles) == 1:
                    answer = next(iter(shared_roles))  # Extract the single shared role
                    
                    # Create question
                    question = f"What role do both {org1} and {org2} have in the agreement?"
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,  
                        'num_set_operations': 1,  # One set intersection operation
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })
        
        return questions
    
    def generate_shared_role_questions_plural(self, graph, doc_num):
        """Generate questions about shared roles between two organizations in an agreement"""
        questions = []
        
        # Get all organizations and their roles
        org_roles = self.get_org_roles(graph)
        
        # Generate all unique combinations of two organizations
        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]
            
            # Ensure each organization has more than one role
            if len(roles1) > 1 and len(roles2) > 1:
                # Find roles shared by both organizations
                shared_roles = roles1 & roles2
                
                # Only proceed if exactly more than role is shared
                if len(shared_roles) > 1:
                    answer = ", ".join(sorted(shared_roles))
                    
                    # Create question
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

    
    def generate_shared_role_subrole_questions(self, graph, doc_num):
        """Generate questions about which single company holds both a role and a subrole in an agreement."""
        questions = []

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)
        print("Extracted organization roles with subroles:", org_roles)  # Debugging

        # Create a mapping of roles to their valid subroles
        role_to_subroles = {}
        for org, roles in org_roles.items():
            for role, subrole in roles:
                if subrole is None:  # Skip None subroles
                    continue
                if role not in role_to_subroles:
                    role_to_subroles[role] = set()
                role_to_subroles[role].add(subrole)

        # Iterate through organizations and their assigned roles
        for org, roles in org_roles.items():
            for role, subrole in roles:
                if subrole is None:  # Skip None subroles
                    continue

                # Ensure this role has subroles in the extracted data
                if role in role_to_subroles and subrole in role_to_subroles[role]:
                    # Check if this specific organization holds both the role and its subrole
                    if (role, subrole) in roles:
                        question = f"What company is both the {subrole} {role} and the {role} in the agreement?"
                        answer = org  # Since we already filtered for a single company
                        
                        print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                        questions.append({
                            'question': question,
                            'answer': answer,
                            'num_hops': 1,
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0  # Single-answer case
                        })

        print("Final generated questions:", questions)  # Debugging
        return questions


    
    def generate_person_position_questions(self, graph, doc_num):
        """Generate questions where only one person holds a specific position in a company."""
        questions = {}

        # Get all organizations with their employees and positions
        org_info = self.get_org_employees_positions(graph)

        # For each organization
        for org_name, info in org_info.items():
            # For each position in the organization
            for position, holders in info['positions'].items():
                if len(holders) == 1:  # Only generate if exactly one person holds the position
                    question = f"Who is the {position} of {org_name}?"
                    answer = list(holders)[0]  # Extract the single person's name

                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0  # Single answer
                    }

        return list(questions.values())
    
    def generate_person_position_questions_multiple(self, graph, doc_num):
        """Generate questions where multiple people hold a specific position in a company."""
        questions = {}

        # Get all organizations with their employees and positions
        org_info = self.get_org_employees_positions(graph)

        # For each organization
        for org_name, info in org_info.items():
            # For each position in the organization
            for position, holders in info['positions'].items():
                if len(holders) > 1:  # Only generate if multiple people hold the position
                    question = f"Who are the {position}s of {org_name}?"
                    answer = ", ".join(sorted(holders))  # Join sorted names for consistency

                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1  # Multiple answers
                    }

        return list(questions.values())

    
    def generate_person_company_role_question(self, graph, doc_num):
            """Generate questions about the role in the agreement of the company where {employee} is employed, only if the company has a single role."""
            questions = {}

            # Get all organizations with their employees and roles
            org_info = self.get_org_employees_and_roles(graph)

            for org_name, info in org_info.items():
                roles = info['roles']  # Extract roles associated with the organization

                if len(roles) == 1:  # Generate only if the company has exactly one role
                    single_role = list(roles)[0]

                    # For each employee in the organization
                    for employee in info['employees']:
                        question = f"What is the role in the agreement of the company where {employee} is employed?"
                        answer = single_role  # Since there's only one role

                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 2,  
                            'num_set_operations': 0,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0  # Single answer
                        }

            return list(questions.values())
    
    def generate_person_company_role_question_multiple(self, graph, doc_num):
        """Generate questions about the roles in the agreement of the company where {employee} is employed, only if the company has multiple roles."""
        questions = {}

        # Get all organizations with their employees and roles
        org_info = self.get_org_employees_and_roles(graph)

        for org_name, info in org_info.items():
            roles = info['roles']  # Extract roles associated with the organization

            if len(roles) > 1:  # Generate only if the company has multiple roles
                sorted_roles = ", ".join(sorted(roles))  # Sort for consistency

                # For each employee in the organization
                for employee in info['employees']:
                    question = f"What are the roles in the agreement of the company where {employee} is employed?"
                    answer = sorted_roles  # Multiple roles as the answer

                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1  # Multiple answers
                    }

        return list(questions.values())

    def generate_location_role_questions(self, graph, doc_num):
        """Generate questions about the role in the agreement of the company based on location, only if the location has a single role."""
        questions = {}

        # Get organization roles and locations
        location_org_roles = self.get_org_roles_and_locations(graph)

        for location, orgs in location_org_roles.items():
            role_set = set()

            for org_name, roles in orgs.items():
                role_set.update(roles)  # Collect all roles associated with this location

            if len(role_set) == 1:  # Only consider if there's exactly one role
                single_role = list(role_set)[0]

                question = f"What is the role in the agreement of the company associated with {location}?"
                answer = single_role  # Since there's only one role

                questions[question] = {
                    'question': question,
                    'answer': answer,
                    'num_hops': 2,  
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0  
                }

        return list(questions.values())
    
    def generate_location_role_questions_multiple(self, graph, doc_num):
        """Generate questions about the roles in the agreement of the company based on location, only if the location has multiple roles."""
        questions = {}

        # Get organization roles and locations
        location_org_roles = self.get_org_roles_and_locations(graph)

        for location, orgs in location_org_roles.items():
            role_set = set()

            for org_name, roles in orgs.items():
                role_set.update(roles)  # Collect all roles associated with this location

            if len(role_set) > 1:  # Only consider if there are multiple roles
                sorted_roles = ", ".join(sorted(role_set))  # Sort roles alphabetically for consistency

                question = f"What are the roles in the agreement of the company associated with {location}?"
                answer = sorted_roles  # Multiple roles as the answer

                questions[question] = {
                    'question': question,
                    'answer': answer,
                    'num_hops': 2,  
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 1  # Multiple roles
                }

        return list(questions.values())


    def generate_location_office_questions(self, graph, doc_num):
        """Generate questions about the location type of an office for a specific organization."""
        questions = {}

        # Get locations and organizations (assuming a function exists that fetches location and organization information)
        location_org_data = self.get_location_org_data(graph)

        print("Location and Organization Data:", location_org_data)  # Debug: print the fetched data

        for location, orgs in location_org_data.items():
            print(f"Processing location: {location}")  # Debug: print the current location being processed

            for org_name, location_types in orgs.items():
                print(f"  Organization: {org_name}, Location Types: {location_types}")  # Debug: print each organization and its location types
                
                # Assuming location_types contains a dictionary like {'location_types': {'BRANCH'}}
                if isinstance(location_types, dict):
                    location_types = location_types.get('location_types', set())  # Extract the actual location types
                    
                # Ensure location_types is a list, not a set, to properly handle each location type
                if isinstance(location_types, set):
                    location_types = list(location_types)  # Convert set to list if necessary

                for location_type in location_types:
                    # Clean the location_type string (strip spaces, lowercase, etc.)
                    location_type_cleaned = location_type.strip().lower()

                    # Generate question for each location type without hardcoding 'location_types'
                    question = f"What is the {location_type_cleaned} office of {org_name}?"

                    # Debug: print the generated question
                    print(f"Generated Question: {question}")

                    # Store the generated question and answer
                    questions[question] = {
                        'question': question,
                        'answer': location,  # Assuming location is the answer here
                        'num_hops': 2,  # Location -> Organization -> Office Location Type
                        'num_set_operations': 0,
                        'document_number': doc_num
                    }

        print("Generated Questions:", questions)  # Debug: print all generated questions at the end

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
                
                # Generate questions for different categories
                # shared_position_questions = self.generate_shared_position_questions(graph, doc_num)
                # shared_position_questions_plural = self.generate_shared_position_questions_plural(graph, doc_num)
                # shared_role_questions = self.generate_shared_role_questions(graph, doc_num)
                # shared_role_questions_plural = self.generate_shared_role_questions_plural(graph, doc_num)
                # shared_role_subrole_questions = self.generate_shared_role_subrole_questions(graph, doc_num)
                # shared_role_subrole_questions_plural = self.generate_shared_role_subrole_questions_plural(graph, doc_num)
                # person_position_questions = self.generate_person_position_questions(graph, doc_num)
                # person_position_questions_multiple = self.generate_person_position_questions_multiple(graph, doc_num)
                # person_company_role_questions = self.generate_person_company_role_question(graph, doc_num)
                # person_company_role_questions_multiple = self.generate_person_company_role_question_multiple(graph, doc_num)
                # location_role_questions = self.generate_location_role_questions(graph, doc_num)
                # location_role_questions_multiple = self.generate_location_role_questions_multiple(graph, doc_num)
                location_office_questions = self.generate_location_office_questions(graph, doc_num)

                # Collecting all questions
                # all_questions.extend(shared_position_questions)
                # all_questions.extend(shared_position_questions_plural)
                # all_questions.extend(shared_role_questions)
                #all_questions.extend(shared_role_questions_plural)
                #all_questions.extend(shared_role_subrole_questions)
                #all_questions.extend(shared_role_subrole_questions_plural)
                # all_questions.extend(person_position_questions)
                # all_questions.extend(person_position_questions_multiple)
                # all_questions.extend(person_company_role_questions)
                # all_questions.extend(person_company_role_questions_multiple)
                # all_questions.extend(location_role_questions)
                # all_questions.extend(location_role_questions_multiple)
                all_questions.extend(location_office_questions)

            except Exception as e:
                print(f"Error processing {ttl_file}: {e}")
        
        # Create DataFrame only if all_questions has data
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
    df.to_csv('qa_dataframe.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe.csv")

if __name__ == "__main__":
    main()
