import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from collections import defaultdict
from typing import Dict, Set
from urllib.parse import unquote


class QuestionGenerator:
    def __init__(self, ttl_dir="extracted_content"):
        self.ttl_dir = ttl_dir
        self.person_name = Namespace("http://example.org/person_name/")
        self.person_position = Namespace("http://example.org/person_position/")
        self.org_name = Namespace("http://example.org/org_name/")
        self.rel = Namespace("http://example.org/relation/")
        self.loc = Namespace("http://example.org/location/")
        self.location_type = Namespace("http://example.org/location_type/")
        self.isInstanceOf = URIRef("http://example.org/isInstanceOf/")
        
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
            person_name = self.clean_uri(person_uri)
            positions = set(self.clean_uri(pos) for pos in position_uris) 
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
            location = self.clean_uri(loc_uri, is_location=True)
            org_name = self.clean_uri(org_uri)
            person_name = self.clean_uri(person_uri)
            position = self.clean_uri(position_uri, is_position=True)
            
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
            raw_location = unquote(str(row.location))
            location = self.clean_uri(raw_location)
            location_map[location][org].add(role)
        
        return location_map

    def _get_org_locations(self, graph):
        """Helper to fetch organization locations with proper SPARQL syntax"""
        org_locations = defaultdict(list)
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?loc; separator="|") AS ?locs)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                 <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?org
        """
        results = graph.query(query)
        for org_uri, locs_str in results:
            org_name = self.clean_uri(org_uri)
            locations = [self.clean_uri(l) for l in locs_str.split('|') if l]
            org_locations[org_name] = locations
        return org_locations
    
    def generate_person_position_questions(self, graph, doc_num):
        """What position does [Person Name] hold?"""
        questions = []
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?person (GROUP_CONCAT(DISTINCT ?position; separator="|") as ?positions)
        WHERE {
            ?person a base:Person ;
                rel:holdsPositionAt ?employment .
            ?employment rel:position ?position .
        }
        GROUP BY ?person
        """
        
        results = graph.query(query)

        for row in results:
            person_uri = row.person
            positions_str = row.positions
            person_name = self.clean_uri(person_uri)
            
            # Split and clean positions
            positions = [self.clean_uri(p) 
                        for p in positions_str.split("|") if p]
            
            if len(positions)==1:
                answer = ", ".join(positions)
                
                questions.append({
                    'question': f"What position does {person_name} hold?",
                    'answer': answer,
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        print(f"Generated {len(questions)} position questions")
        return questions

    
    def generate_person_organization_questions(self, graph, doc_num):
        '''In what organization does [Person Name] work?'''
        questions = []
        query = """
        PREFIX base: <http://example.org/base/>
        PREFIX rel: <http://example.org/relation/>
        
        SELECT DISTINCT ?person ?org
        WHERE {
            ?person a base:Person ;
                rel:holdsPositionAt ?employment .  # Navigate through employment node
            ?employment rel:organization ?org .      # Get organization via employment
        }
        """
        
        org_mapping = defaultdict(set)
        # First collect all organizations per person
        for row in graph.query(query):
            person_uri = row.person
            org_uri = row.org
            org_mapping[person_uri].add(org_uri)
        
        # Generate questions with proper answer handling
        for person_uri, org_uris in org_mapping.items():
            person_name = self.clean_uri(person_uri)
            orgs = [self.clean_uri(org) for org in org_uris]
            if len(orgs)==1: 
                questions.append({
                    'question': f"In what organization does {person_name} work?",
                    'answer': ", ".join(orgs) if len(orgs) > 1 else orgs[0],
                    'num_hops': 1,  
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension':0
                })
            
        return questions
    
    def generate_organization_representative_questions(self, graph, doc_num):
        '''Who is the representative of [Org Name]? [if one]'''
        questions = []
        query = """
        SELECT ?org (SAMPLE(?person) AS ?rep)
        WHERE {
            ?person <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?org
        HAVING (COUNT(?person) = 1)
        """
        for org_uri, rep_uri in graph.query(query):
            questions.append({
                'question': f"Who is the representative of {self.clean_uri(org_uri)}?",
                'answer': self.clean_uri(rep_uri),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })
        return questions
    
    def generate_org_location_questions(self, graph, doc_num):
        '''What is the location of [Organization Name] [if one]?'''
        questions = []
        for org, locs in self._get_org_locations(graph).items():
            if len(locs) == 1:
                questions.append({
                    'question': f"What is the location of {org}?",
                    'answer': locs[0],
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })
        return questions
    
    def generate_location_company_questions(self, graph, doc_num):
        '''Which company is associated with [Location]?'''
        questions = []
        query = """
        SELECT ?loc (GROUP_CONCAT(?org; separator="|") AS ?orgs)
        WHERE {
            ?loc a <http://example.org/base/Location> .
            ?org <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?loc
        HAVING (COUNT(?org) = 1)
        """
        for loc_uri, orgs_str in graph.query(query):
            loc_name = self.clean_uri(loc_uri)
            org_name = self.clean_uri(orgs_str.split('|')[0])
            questions.append({
                'question': f"Which company is associated with {loc_name}?",
                'answer': org_name,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })
        return questions
    
    def generate_location_type_questions(self, graph, doc_num):
        '''What type of location is [Location] (e.g., Headquarters, Trade Operations, etc.)? [if one]'''
        questions = []
        query = """
        SELECT ?loc (GROUP_CONCAT(?type; separator="|") AS ?types)
        WHERE {
            ?loc a <http://example.org/base/Location> ;
                 <http://example.org/isInstanceOf/> ?type .
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?loc
        HAVING (COUNT(?type) = 1)
        """
        for loc_uri, types_str in graph.query(query):
            questions.append({
                'question': f"What type of location is {self.clean_uri(loc_uri)}?",
                'answer': self.clean_uri(types_str.split('|')[0]),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })
        return questions
    
    def generate_org_role_questions(self, graph, doc_num):
        """What is the role of {Org Name} in the agreement?"""
        questions = []
        org_roles = self.get_org_roles(graph)
        for org_name, roles in org_roles.items():
            for role in roles:
                if len(roles)==1:
                    questions.append({
                        'question': f"What is the role of {org_name} in the agreement?",
                        'answer': role,
                        'num_hops': 1,
                        'num_set_operations': 0,
                        'document_number': doc_num,
                        'multiple_answer_dimension': 0
                    })

        return questions



    def generate_company_by_role_questions(self, graph, doc_num):
        """What company is the [Org Role (+ Sub‑Role)] in the agreement?"""
        questions = []

        # Get the org→roles mapping
        org_roles = self.get_org_roles(graph)  # { org_name: set(role_strings) }

        # Invert to role→{org_names}
        role_to_org = defaultdict(set)
        for org_name, roles in org_roles.items():
            for role in roles:
                role_to_org[role].add(org_name)

        # Generate a question only when a role has exactly one org
        for role, orgs in role_to_org.items():
            if len(orgs) == 1:
                org_name = next(iter(orgs))
                questions.append({
                    'question': f"What company is the {role} in the agreement?",
                    'answer': org_name,
                    'num_hops': 1,
                    'num_set_operations': 0,
                    'document_number': doc_num,
                    'multiple_answer_dimension': 0
                })

        return questions
        
    def generate_all_questions(self):
        all_questions = []
        ttl_files = [f for f in glob.glob(os.path.join(self.ttl_dir, "*.ttl")) if not f.endswith("ontology.ttl")]
        for ttl_file in ttl_files:
            try:
                doc_num = os.path.splitext(os.path.basename(ttl_file))[0]
                graph = self.load_graph(ttl_file)
                all_questions.extend(self.generate_person_position_questions(graph, doc_num))
                all_questions.extend(self.generate_person_organization_questions(graph, doc_num))
                all_questions.extend(self.generate_organization_representative_questions(graph, doc_num))
                all_questions.extend(self.generate_org_location_questions(graph, doc_num))
                all_questions.extend(self.generate_location_company_questions(graph, doc_num))
                all_questions.extend(self.generate_location_type_questions(graph, doc_num))
                all_questions.extend(self.generate_org_role_questions(graph, doc_num))
                all_questions.extend(self.generate_company_by_role_questions(graph, doc_num))
            except Exception as e:
                print(f"Error processing {ttl_file}: {e}")
        df = pd.DataFrame(all_questions)
        cols = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations', 'multiple_answer_dimension']
        return df[cols]

if __name__ == "__main__":
    generator = QuestionGenerator()
    df = generator.generate_all_questions()
    df.to_csv('level1.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe.csv")
