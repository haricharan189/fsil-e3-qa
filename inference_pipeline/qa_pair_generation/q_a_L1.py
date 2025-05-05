import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os
from collections import defaultdict

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
    
    def get_org_role_subrole_map(self, graph):
        """Get mapping of org_roles to their sub_roles"""
        role_subrole_map = {}
        
        query = """
        SELECT DISTINCT ?role ?sub_role
        WHERE {
            ?sub_role rdfs:subClassOf ?role .
            FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
        }
        """
        results = graph.query(query)
        
        for role_uri, sub_role_uri in results:
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri)
            role_subrole_map.setdefault(role, set()).add(sub_role)
        return role_subrole_map
    
    def get_org_roles_with_subroles(self, graph):
        """Get organizations with normalized roles in 'subrole role' format"""
        org_roles = defaultdict(lambda: defaultdict(set))
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?org ?role (GROUP_CONCAT(?parentRole; separator="|") AS ?parents)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?role .
            OPTIONAL {
                ?role rdfs:subClassOf ?parentRole .
                FILTER(STRSTARTS(STR(?parentRole), "http://example.org/org_role/"))
            }
        }
        GROUP BY ?org ?role
        """
        results = list(graph.query(query))
        covered_parents = defaultdict(set)

        # First pass: subroles
        for org_uri, role_uri, parents_str in results:
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            parents = [self.clean_uri(p) for p in parents_str.split('|') if p]
            if str(role_uri).startswith("http://example.org/org_sub_role/"):
                for parent in parents:
                    subrole = role.title()
                    parent_role = parent.title()
                    org_roles[org_name]['roles'].add(f"{subrole} {parent_role}")
                    covered_parents[org_name].add(parent_role)

        # Second pass: main roles
        for org_uri, role_uri, _ in results:
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(role_uri)
            if str(role_uri).startswith("http://example.org/org_sub_role/"):
                continue
            main_role = role.title()
            if main_role not in covered_parents[org_name]:
                org_roles[org_name]['roles'].add(main_role)
        return org_roles
    
    def get_org_employees_and_roles(self, graph):
        """Fetch all organizations, their employees, and their roles"""
        org_info = {}
        query = """
            SELECT DISTINCT ?org ?employee ?orgRole ?orgSubRole
            WHERE {
                ?org a <http://example.org/base/Organization> ;
                     <http://example.org/isInstanceOf/> ?orgRole .
                OPTIONAL { ?org <http://example.org/relation/hasEmployee> ?employee . }
                OPTIONAL {
                    ?orgRole rdfs:subClassOf ?orgSubRole .
                    FILTER(STRSTARTS(STR(?orgSubRole), "http://example.org/org_role/"))
                }
                FILTER(STRSTARTS(STR(?orgRole), "http://example.org/org_role/") ||
                       STRSTARTS(STR(?orgSubRole), "http://example.org/org_role/"))
            }
        """
        results = graph.query(query)
        for org_uri, emp_uri, role_uri, sub_uri in results:
            org_name = self.clean_uri(org_uri)
            employee = self.clean_uri(emp_uri) if emp_uri else None
            role = self.clean_uri(role_uri)
            sub = self.clean_uri(sub_uri) if sub_uri else None
            final = f"{role} {sub}".strip() if sub else role
            ent = org_info.setdefault(org_name, {'employees': set(), 'roles': set()})
            ent['roles'].add(final)
            if employee:
                ent['employees'].add(employee)
        return org_info
    
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
        '''What is the position of [Person Name]? [if one]'''
        questions = []
        query = """
        SELECT DISTINCT ?person (SAMPLE(?position) AS ?pos)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                     <http://example.org/isInstanceOf/> ?position .
            FILTER(STRSTARTS(STR(?position), "http://example.org/person_position/"))
        }
        GROUP BY ?person
        HAVING (COUNT(?position) = 1)
        """
        for person_uri, pos_uri in graph.query(query):
            questions.append({
                'question': f"What is the position of {self.clean_uri(person_uri)}?",
                'answer': self.clean_uri(pos_uri),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })
        return questions
    
    def generate_person_organization_questions(self, graph, doc_num):
        '''In what organization does [Person Name] work?'''
        questions = []
        query = """
        SELECT ?person (SAMPLE(?org) AS ?organization)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                    <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?person
        HAVING (COUNT(?org) = 1)
        """
        for person_uri, org_uri in graph.query(query):
            questions.append({
                'question': f"In what organization does {self.clean_uri(person_uri)} work?",
                'answer': self.clean_uri(org_uri),
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
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
        '''What is the location of [Org Name]? [if one]'''
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
        '''What is the role of [Org Name] in the agreement? [if one]'''
        questions = []
        role_subrole_map = self.get_org_roles_with_subroles(graph)
        query = """
        SELECT ?org (GROUP_CONCAT(?role; separator="|") AS ?roles)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                 <http://example.org/isInstanceOf/> ?role .
            { FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/")) }
            UNION
            { FILTER(STRSTARTS(STR(?role), "http://example.org/org_sub_role/")) }
        }
        GROUP BY ?org
        HAVING (COUNT(?role) = 1)
        """
        for org_uri, roles_str in graph.query(query):
            org_name = self.clean_uri(org_uri)
            role = self.clean_uri(roles_str.split('|')[0])
            answer = f"{role} ({' '.join(role_subrole_map.get(role, []))})" if role in role_subrole_map else role
            questions.append({
                'question': f"What is the role of {org_name} in the agreement?",
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answer_dimension': 0
            })
        return questions
    
    def generate_company_by_role_questions(self, graph, doc_num):
        '''What company is the [Org Role (+ Sub-Role)] in the agreement? [if one]'''
        questions = []
        org_roles = self.get_org_roles_with_subroles(graph)
        role_org_map = defaultdict(set)
        for org, data in org_roles.items():
            for role in data['roles']:
                role_org_map[role].add(org)
        for role, orgs in role_org_map.items():
            if len(orgs) == 1:
                questions.append({
                    'question': f"What company is the {role} in the agreement?",
                    'answer': next(iter(orgs)),
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
