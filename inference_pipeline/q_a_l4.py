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

                # Filter to include only roles that start with the org_role or org_sub_role
                FILTER(STRSTARTS(STR(?finalRole), "http://example.org/org_role/") ||
                    STRSTARTS(STR(?finalRole), "http://example.org/org_sub_role/"))
            }
        """

        results = graph.query(query)

        for row in results:
            print("Processing row: ", row)  # Debugging step
            org_uri, role_or_subrole_uri, parent_role_uri = row
            org_name = self.clean_uri(org_uri)
            role_or_subrole = self.clean_uri(role_or_subrole_uri)
            parent_role = self.clean_uri(parent_role_uri) if parent_role_uri else None

            if org_name not in org_roles:
                org_roles[org_name] = set()

            # If there's a parent role (subrole exists), associate the subrole with the parent role
            if parent_role:
                org_roles[org_name].add((parent_role, role_or_subrole))
            else:
                # No parent role, just the role
                org_roles[org_name].add((role_or_subrole, None))

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

    
    def generate_position_comparison_questions(self, graph, doc_num):
        """Generate questions comparing positions between one person and two others, ensuring they share at least one position."""
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)

        # Filter out people who have at least one position
        valid_people = {p: pos for p, pos in person_positions.items() if len(pos) > 1}

        # Generate all unique combinations of three people
        for person1, person2, person3 in combinations(valid_people.keys(), 3):
            # Get positions for each person
            positions1 = valid_people[person1]
            positions2 = valid_people[person2]
            positions3 = valid_people[person3]

            # Find positions shared by at least one of them
            shared_positions = positions1 & positions2 | positions1 & positions3 | positions2 & positions3

            # Proceed only if they share at least one position
            if shared_positions:
                # Find positions unique to person1 (not held by person2 or person3)
                unique_positions = positions1 - (positions2 | positions3)

                # Only proceed if exactly one unique position exists
                if len(unique_positions) == 1:
                    unique_position = next(iter(unique_positions))

                    # Create the question
                    question = f"What is the position held by {person1} but not by {person2} or {person3}?"
                    answer = unique_position

                    print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1, 
                        'num_set_operations': 3, 
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })

        print("Final generated questions:", questions)  # Debugging
        return questions

    
    def generate_shared_position_exclusion_questions(self, graph, doc_num):
        """Generate questions about shared positions between two people that are not held by a third person."""
        questions = []
        
        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)

        # Filter people who have at least one position
        valid_people = {p: pos for p, pos in person_positions.items() if len(pos) > 0}

        # Generate all unique combinations of three people
        for person1, person2, person3 in combinations(valid_people.keys(), 3):
            # Get positions for each person
            positions1 = valid_people[person1]
            positions2 = valid_people[person2]
            positions3 = valid_people[person3]

            # Find positions shared by all three people
            shared_by_all = positions1 & positions2 & positions3

            # Only proceed if there is exactly one shared position among the three
            if len(shared_by_all) == 1:
                shared_position = next(iter(shared_by_all))

                # Create the question
                question = f"What is the position held by {person1} and {person2} but not by {person3}?"
                answer = shared_position

                print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                questions.append({
                    'question': question,
                    'answer': answer,
                    'num_hops': 1,  # Need to check positions of all three people
                    'num_set_operations': 3,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        print("Final generated questions:", questions)  # Debugging
        return questions
    

    def generate_exclusive_shared_role_questions(self, graph, doc_num):
        """Generate questions about roles shared by two organizations but not held by a third organization in an agreement."""
        questions = []

        # Get all organizations and their roles
        org_roles = self.get_org_roles(graph)

        # Filter organizations that have more than one role
        valid_orgs = {org: roles for org, roles in org_roles.items() if len(roles) > 1}

        # Generate all unique combinations of three organizations
        for org1, org2, org3 in combinations(valid_orgs.keys(), 3):
            roles1 = valid_orgs[org1]
            roles2 = valid_orgs[org2]
            roles3 = valid_orgs[org3]

            # Find roles common to all three organizations
            common_roles_all = roles1 & roles2 & roles3

            # Ensure all three organizations share more than one role
            if len(common_roles_all) > 1:
                # Find roles shared by org1 and org2 but NOT by org3
                exclusive_shared_roles = (roles1 & roles2) - roles3

                # Proceed only if exactly one exclusive shared role exists
                if len(exclusive_shared_roles) == 1:
                    exclusive_role = next(iter(exclusive_shared_roles))

                    # Create the question
                    question = f"What role do {org1} and {org2} have in the agreement which is not the role of {org3}?"
                    answer = exclusive_role

                    print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 1,  
                        'num_set_operations': 3,  
                        'document_number': doc_num,
                        'multiple_answers_dimension': 0
                    })

        print("Final generated questions:", questions)  # Debugging
        return questions
    
    def generate_shared_role_subrole_questions(self, graph, doc_num):
        """Generate questions about which companies hold two specific roles and subroles but not a third in an agreement."""
        questions = []

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)
        print("Extracted organization roles with subroles:", org_roles)  # Debugging

        # Remove (role, subrole) pairs where subrole is None or "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.strip().lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Create a mapping of roles to their subroles
        role_to_subroles = {}
        for org, roles in cleaned_org_roles.items():
            for role, subrole in roles:
                if role not in role_to_subroles:
                    role_to_subroles[role] = set()
                role_to_subroles[role].add(subrole)

        # Process organizations with multiple role-subrole pairs
        orgs_with_multiple_roles = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) > 1}

        for org, roles in orgs_with_multiple_roles.items():
            role_subrole_list = list(roles)

            for i, (role_1, subrole_1) in enumerate(role_subrole_list):
                for j, (role_2, subrole_2) in enumerate(role_subrole_list):
                    if j <= i or (role_1, subrole_1) == (role_2, subrole_2):
                        continue  # Skip identical or duplicate role-subrole pairs

                    # Ensure these role-subrole pairs share at least one role-subrole in common
                    common_subroles = set(role_to_subroles.get(role_1, [])) & set(role_to_subroles.get(role_2, []))
                    
                    if not common_subroles:
                        continue  # Skip if no common subroles

                    # Now check for the third role-subrole pair
                    for k, (role_3, subrole_3) in enumerate(role_subrole_list):
                        if k == i or k == j:
                            continue  # Skip the already selected pairs

                        # Find organizations that hold both (role_1, subrole_1) and (role_2, subrole_2)
                        valid_orgs = {org for org, org_roles in cleaned_org_roles.items() 
                                    if (role_1, subrole_1) in org_roles and (role_2, subrole_2) in org_roles}
                        
                        # Filter out organizations that also have (role_3, subrole_3)
                        valid_orgs = {org for org in valid_orgs if (role_3, subrole_3) not in org_roles.get(org, [])}

                        # Generate question only if exactly one organization meets the criteria
                        if len(valid_orgs) == 1:
                            valid_org = list(valid_orgs)[0]
                            question = (f"What company is the {subrole_1} {role_1}, and the {subrole_2} {role_2} "
                                        f"but not the {subrole_3} {role_3} in the agreement?")
                            answer = valid_org
                            
                            print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                            questions.append({
                                'question': question,
                                'answer': answer,
                                'num_hops': 1,
                                'num_set_operations': 3,
                                'document_number': doc_num,
                                'multiple_answer_dimension': 0
                            })

        print("Final generated questions:", questions)  # Debugging
        return questions


    def generate_exclusive_role_subrole_questions(self, graph, doc_num):
        """Generate questions about companies that hold one specific role and subrole but not two others in an agreement."""
        questions = []

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)
        print("Extracted organization roles with subroles:", org_roles)  # Debugging

        # Remove (role, subrole) pairs where subrole is None or "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.strip().lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Process organizations with multiple role-subrole pairs
        orgs_with_multiple_roles = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) > 1}

        for org, roles in orgs_with_multiple_roles.items():
            role_subrole_list = list(roles)

            # Generate combinations of roles and subroles
            for i, (role_1, subrole_1) in enumerate(role_subrole_list):
                for j, (role_2, subrole_2) in enumerate(role_subrole_list):
                    if j == i:
                        continue  # Skip identical pairs

                    for k, (role_3, subrole_3) in enumerate(role_subrole_list):
                        if k == i or k == j:
                            continue  # Skip already selected pairs

                        # Now, filter organizations based on (role_1, subrole_1) but exclude (role_2, subrole_2) and (role_3, subrole_3)
                        valid_orgs = {org for org, org_roles in cleaned_org_roles.items() 
                                    if (role_1, subrole_1) in org_roles and 
                                    (role_2, subrole_2) not in org_roles and 
                                    (role_3, subrole_3) not in org_roles}

                        # Generate question only if exactly one organization meets the criteria
                        if len(valid_orgs) == 1:
                            valid_org = list(valid_orgs)[0]
                            question = (f"What company is the {subrole_1} {role_1} but not the {subrole_2} {role_2} "
                                        f"or the {subrole_3} {role_3} in the agreement?")
                            answer = valid_org
                            
                            print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                            questions.append({
                                'question': question,
                                'answer': answer,
                                'num_hops': 1,
                                'num_set_operations': 3,
                                'document_number': doc_num,
                                'multiple_answer_dimension': 0
                            })

        print("Final generated questions:", questions)  # Debugging
        return questions



    
    def generate_person_position_questions_for_unique_dual_role(self, graph, doc_num):
        """Generate questions about positions in companies that hold two specific roles in an agreement."""
        questions = {}  

        # Get all organizations with their employees and positions
        org_info = self.get_org_employees_positions(graph)
        
        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)

        # Remove (role, subrole) pairs where subrole is "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Identify companies that hold exactly two distinct roles/subroles
        dual_role_companies = {}
        for org, roles in cleaned_org_roles.items():
            if len(roles) == 2:  # Ensure exactly two roles
                dual_role_companies[org] = roles

        # Process each uniquely identified dual-role company
        for org_name, roles in dual_role_companies.items():
            if org_name in org_info:  # Ensure the company has known positions
                for position, holders in org_info[org_name]['positions'].items():
                    if len(holders) == 1:  # Ensure the position has exactly one person
                        role_1, subrole_1 = list(roles)[0]
                        role_2, subrole_2 = list(roles)[1]

                        # Format role-subrole pairs properly
                        role_1_desc = f"{subrole_1} {role_1}" if subrole_1 else role_1
                        role_2_desc = f"{subrole_2} {role_2}" if subrole_2 else role_2

                        # Construct question
                        question = (f"Who is the {position} of the company which is both the {role_1_desc} "
                                    f"and the {role_2_desc} in the agreement?")
                        
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
                            'multiple_answer_dimension':0
                        }
        
        return list(questions.values())
    
    def generate_dual_person_position_questions(self, graph, doc_num):
        """Generate questions about a person who holds two positions in a company that has a specific role in an agreement."""
        questions = {}

        # Get all persons and their positions
        person_positions = self.get_person_positions(graph)

        # Get all organizations and their people with positions
        org_people_positions = self.get_org_people_positions(graph)

        # Get all organizations and their (role, subrole) pairs
        org_roles = self.get_org_roles_with_subroles(graph)

        # Remove (role, subrole) pairs where subrole is "None"
        cleaned_org_roles = {
            org: {(role, subrole) for role, subrole in roles if subrole and subrole.lower() != "none"}
            for org, roles in org_roles.items()
        }

        # Remove organizations that have no remaining roles
        cleaned_org_roles = {org: roles for org, roles in cleaned_org_roles.items() if roles}

        # Identify companies with a **single, uniquely identifiable role**
        uniquely_identified_companies = {org: roles for org, roles in cleaned_org_roles.items() if len(roles) == 1}

        # Process each uniquely identified company
        for org_name, roles in uniquely_identified_companies.items():
            if org_name in org_people_positions:  # Ensure the company has known employees
                # Iterate through each person in the organization
                for person, positions in org_people_positions[org_name].items():
                    if len(positions) == 2:
                        position_1, position_2 = sorted(positions)  # Ensure consistent ordering
                        role, subrole = list(roles)[0]  # Extract the unique role

                        # Format role-subrole pair properly
                        role_desc = f"{subrole} {role}" if subrole else role

                        # Construct question
                        question = (f"Who is both the {position_1} and {position_2} of the company "
                                    f"which is the {role_desc} in the agreement?")
                        
                        answer = person

                        print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                        # Store the question and answer
                        questions[question] = {
                            'question': question,
                            'answer': answer,
                            'num_hops': 3,  # Organization -> Role -> Positions -> Position Holder
                            'num_set_operations': 1,
                            'document_number': doc_num,
                            'multiple_answer_dimension': 0
                        }
        
        return list(questions.values())
    
    def generate_dual_person_position_location_questions(self, graph, doc_num):
        """Generate questions about a person holding two positions in a company associated with a specific location."""
        questions = {}

        # Get all organizations and their people with positions
        org_people_positions = self.get_org_people_positions(graph)
        print("Extracted organization people and positions:", org_people_positions)  # Debugging

        # Get all organizations and their associated locations
        location_org_data = self.get_org_roles_and_locations(graph)
        print("Extracted location-organization mapping:", location_org_data)  # Debugging

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
                            
                            # Answer is the single known position holder
                            answer = person

                            print(f"Generated question: {question} | Answer: {answer}")  # Debugging

                            # Store the question and answer
                            questions[question] = {
                                'question': question,
                                'answer': answer,
                                'num_hops': 3,  # Location -> Organization -> Positions -> Position Holder
                                'num_set_operations': 1,
                                'document_number': doc_num,
                                'multiple_answer_dimension':0
                            }
        
        print("Final generated questions:", questions)  # Debugging
        return list(questions.values())

    def generate_dual_position_questions(self, graph, doc_num):
        """Generate questions about individuals holding multiple positions in a uniquely identifiable company."""
        questions = {}

        # Extract employment data: Organizations and their employees with positions
        org_people_positions = self.get_org_people_positions(graph)

        print("Extracted organization people and positions:", org_people_positions)  # Debugging

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
        """Generate questions about organizations holding multiple roles (with or without subroles) at a specific location."""
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

           
        print("Final generated questions:", questions)  # Debugging
        return list(questions.values())

    def generate_all_questions(self):
        """Generate questions from all TTL files"""
        all_questions = []
        
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) 
                    if not f.endswith("ontology.ttl")]
        
        for ttl_file in ttl_files:
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                graph = self.load_graph(ttl_file)
                
                shared_position_questions = self.generate_position_comparison_questions(graph, doc_num)
                shared_position_exclusion_questions = self.generate_shared_position_exclusion_questions(graph, doc_num)
                shared_role_exclusion_questions = self.generate_exclusive_shared_role_questions(graph, doc_num)
                shared_role_subrole_questions = self.generate_shared_role_subrole_questions(graph, doc_num)
                shared_role_subrole_excusion_questions = self.generate_exclusive_role_subrole_questions(graph, doc_num)
                person_position_questions_for_unique_dual_role = self.generate_person_position_questions_for_unique_dual_role(graph, doc_num)
                dual_person_position_questions = self.generate_dual_person_position_questions(graph, doc_num)
                dual_person_position_location_questions = self.generate_dual_person_position_location_questions(graph, doc_num)
                dual_position_questions = self.generate_dual_position_questions(graph, doc_num)
                location_dual_role_subrole_questions = self.generate_location_dual_role_subrole_questions(graph, doc_num)
            
                all_questions.extend(shared_position_questions)
                all_questions.extend(shared_position_exclusion_questions)
                all_questions.extend(shared_role_exclusion_questions)
                all_questions.extend(shared_role_subrole_questions)
                all_questions.extend(shared_role_subrole_excusion_questions)
                all_questions.extend(person_position_questions_for_unique_dual_role)
                all_questions.extend(dual_person_position_questions)
                all_questions.extend(dual_person_position_location_questions)
                all_questions.extend(dual_position_questions)
                all_questions.extend(location_dual_role_subrole_questions)
                
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
        df = pd.DataFrame(all_questions, columns = columns_order).fillna("")
        
        return df
   
    

def main():
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level4.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe.csv")
    
    print("\nSample questions:")
    print("\nPosition questions:")
    print(df[df['question'].str.startswith('What is the position')].head(2))
    print("\nRole questions:")
    print(df[df['question'].str.startswith('What is the role')].head(2))

if __name__ == "__main__":
    main()
