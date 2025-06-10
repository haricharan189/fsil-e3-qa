import pandas as pd
from rdflib import Graph, Namespace, URIRef
from urllib.parse import unquote
import glob
import os
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

        # SPARQL query for reified employment structure
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

    def generate_shared_position_questions_plural(self, graph, doc_num):
        """[LEVEL 3] What are the positions held by both [Person Name 1] and [Person Name 2]?"""
        questions = []
        person_positions = self.get_person_positions(graph)
        
        # Generate all unique pairs of people
        for person1, person2 in combinations(person_positions.keys(), 2):
            positions1 = person_positions[person1]
            positions2 = person_positions[person2]
            shared_positions = positions1 & positions2 

            if len(shared_positions) > 1:  
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
        org_roles = self.get_org_roles(graph) 
        
        # Create inverse mapping: {frozenset(roles): [orgs]}
        role_pairs_to_orgs = defaultdict(list)
        for org, roles in org_roles.items():
            if len(roles) > 1:
                for role_pair in combinations(roles, 2):
                    role_pairs_to_orgs[frozenset(role_pair)].append(org)
        
        # Generate questions for each role pair with multiple orgs
        for role_pair, orgs in role_pairs_to_orgs.items():
            if len(orgs) >= 1:  
                role_names = [self.clean_uri(r) for r in role_pair]
                question = (
                    f"What companies are both the {role_names[0]} "
                    f"and {role_names[1]} in the agreement?"
                )
                org_names = sorted([self.clean_uri(o) for o in orgs])
                questions.append({
                    'question': question,
                    'answer': ", ".join(org_names),
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

            # Only if both have >1 positions and exactly one unique
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
        org_roles = self.get_org_roles(graph)  
        for org1, org2 in combinations(org_roles.keys(), 2):
            roles1 = org_roles[org1]
            roles2 = org_roles[org2]

            if not (roles1 & roles2):
                continue


            unique_to_org1 = roles1 - roles2
            unique_to_org2 = roles2 - roles1

            # Clean names (if org1/org2 are URIs; if they're already cleaned names, you can skip)
            org1_name = self.clean_uri(org1)
            org2_name = self.clean_uri(org2)

            if len(unique_to_org1) == 1:
                role = next(iter(unique_to_org1))
                questions.append({
                    'question': (
                        f"What role does {org1_name} have in the agreement "
                        f"which is not the role of {org2_name}?"
                    ),
                    'answer': role,
                    'num_hops': 2,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })


            if len(unique_to_org2) == 1:
                role = next(iter(unique_to_org2))
                questions.append({
                    'question': (
                        f"What role does {org2_name} have in the agreement "
                        f"which is not the role of {org1_name}?"
                    ),
                    'answer': role,
                    'num_hops': 2,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions

    def generate_position_organization_questions_plural(self, graph, doc_num):
        """[Level 3] Who are the [Position]s of [Organization]?"""
        questions = []
        org_info = self.get_org_employee_positions(graph)
        for org_name, info in org_info.items():
            for position, holders in info['positions'].items():
                if len(holders) > 1:  
                    question = f"Who are the {position}s of {org_name}?"
                    print(org_name)
                    answer = ", ".join(sorted(holders))  

                    questions.append({
                        'question': question,
                        'answer': answer,
                        'num_hops': 2,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 1  
                    })

        return questions
    
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
            org_info = self.get_org_employee_positions(graph)
            
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
        org_info = self.get_org_employee_positions(graph)
        
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
        What company is the [Org Role (+ Sub‑Role) 1] but not the [Org Role (+ Sub‑Role) 2] in the agreement? [if one]
        """
        questions = []
        org_roles = self.get_org_roles(graph)  
        
        # Invert to create role→{org_name}
        role_to_orgs = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_orgs[role].add(org_name)
        
        # For every unordered pair of roles
        for role1, role2 in combinations(role_to_orgs, 2):
            orgs1 = role_to_orgs[role1]
            orgs2 = role_to_orgs[role2]
            
            # 1) Skip if they have no company in common—no contrast possible
            common_orgs = orgs1 & orgs2
            if not common_orgs:
                continue

            diff = orgs1 - orgs2
            if len(diff) == 1:
                org_only = next(iter(diff))
                questions.append({
                    'question': (
                        f"What company is the {role1} "
                        f"but not the {role2} in the agreement?"
                    ),
                    'answer': org_only,
                    'num_hops': 2,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })
            
            diff_rev = orgs2 - orgs1
            if len(diff_rev) == 1:
                org_only_rev = next(iter(diff_rev))
                questions.append({
                    'question': (
                        f"What company is the {role2} "
                        f"but not the {role1} in the agreement?"
                    ),
                    'answer': org_only_rev,
                    'num_hops': 2,
                    'num_set_operations': 1,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions
    
    def generate_person_position_by_org_role_questions(self, graph, doc_num):
        """
        "Who is the [Position] of the company which is the [Org Role] in the agreement?"
        """
        questions = []
        
        # 1) Get organization→roles map
        org_roles = self.get_org_roles(graph)  

        # 2) Invert to role→set(org_name)
        role_to_org = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_org[role].add(org_name)
        
        # 3) Get organization→positions→holders
        org_positions = self.get_org_employee_positions(graph)  

        # 4) Build your questions
        for role, orgs in role_to_org.items():
            # only if exactly one company has this role
            if len(orgs) != 1:
                continue
            org_name = next(iter(orgs))
            
            # fetch positions dict for that org
            positions_dict = org_positions.get(org_name, {}).get('positions', {})
            
            # find positions that have exactly one holder
            singleton = {
                pos: next(iter(holders))
                for pos, holders in positions_dict.items()
                if len(holders) == 1
            }
            # only if exactly one such position
            if len(singleton) != 1:
                continue
            
            position, person = next(iter(singleton.items()))
            questions.append({
                'question': f"Who is the {position} of the company which is the {role} in the agreement?",
                'answer': person,
                'num_hops': 2,
                'num_set_operations': 1,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })

        return questions
    
    def generate_location_type_by_org_role_questions(self, graph, doc_num):
        """
        What is the address of [Location Type] of the company which is the [Org Role (+ Sub‑Role)] in the agreement?
        """
        questions = []

        # Build role(+subrole) → {org_name} map
        org_roles = self.get_org_roles(graph)
        role_to_org = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_org[role].add(org_name)

        # Fetch org_name → location_address and location_type via SPARQL
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel:  <http://example.org/relation/>
        
        SELECT DISTINCT ?org (GROUP_CONCAT(?loc; separator="|") AS ?locs) (SAMPLE(?type) AS ?loc_type)
        WHERE {
            ?org a base:Organization ;
                rel:hasLocationAt ?loc .
            ?loc <http://example.org/isInstanceOf/> ?type .
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?org
        HAVING (COUNT(DISTINCT ?type) = 1)
        """
        loc_results = graph.query(query)

        org_to_loc_info = {}
        for org_uri, locs_concat, loc_type_uri in loc_results:
            org_name = self.clean_uri(org_uri)
            loc_uris = locs_concat.split("|")
            loc_address = self.clean_uri(loc_uris[0])  # Clean first location URI to get address
            loc_type = self.clean_uri(loc_type_uri)
            org_to_loc_info[org_name] = (loc_type, loc_address)

        # Generate questions
        for role, orgs in role_to_org.items():
            if len(orgs) != 1:
                continue
            org_name = next(iter(orgs))
            if org_name not in org_to_loc_info:
                continue

            loc_type, loc_address = org_to_loc_info[org_name]
            q_text = (
                f"What is the address of {loc_type} of the company which is the {role} in the agreement?"
            )
            questions.append({
                'question': q_text,
                'answer': loc_address,  # Now returns the cleaned location address
                'num_hops': 3,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })

        return questions
    
    def generate_address_of_location_type_by_person_questions(self, graph, doc_num):
        """
        What is the address of [Location Type] of the company where [Person Name] is employed?
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
                # all_questions.extend(self.generate_shared_position_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_shared_role_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_shared_role_subrole_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_position_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_org_role_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_org_role_subrole_comparison_questions(graph, doc_num))
                # all_questions.extend(self.generate_position_organization_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_person_company_role_question_plural(graph, doc_num))
                # all_questions.extend(self.generate_location_role_questions_plural(graph, doc_num))
                # all_questions.extend(self.generate_position_organization_questions_pair(graph, doc_num))
                # all_questions.extend(self.generate_person_position_by_org_role_questions(graph, doc_num))
                # all_questions.extend(self.generate_location_position_questions(graph, doc_num))
                # all_questions.extend(self.generate_employee_org_position_questions(graph, doc_num))
                # all_questions.extend(self.generate_location_type_by_org_role_questions(graph, doc_num))
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
