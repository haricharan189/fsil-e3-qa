import pandas as pd
from rdflib import Graph, Namespace, URIRef
import glob
import os

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
    
    def generate_position_questions(self, graph, doc_num):
        """Generate questions about person positions"""
        questions = []
        
        # Modified SPARQL query to group positions by person
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
            positions = [self.clean_uri(pos) for pos in position_uris]
            
            # Join multiple positions with commas
            answer = ", ".join(positions)
            
            # Create question with appropriate template based on number of positions
            has_multiple = len(positions) > 1
            question = f"What are the positions of {person_name}?" if has_multiple else f"What is the position of {person_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_person_org_questions(self, graph, doc_num):
        """Generate questions about where people work"""
        questions = []
        
        # SPARQL query to get all persons and their organizations
        query = """
        SELECT DISTINCT ?person (GROUP_CONCAT(?org; separator="|") as ?orgs)
        WHERE {
            ?person a <http://example.org/base/Person> ;
                   <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?person
        """
        
        results = graph.query(query)
        
        for row in results:
            person_uri = row[0]
            org_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            person_name = self.clean_uri(person_uri)
            orgs = [self.clean_uri(org) for org in org_uris]
            
            # Join multiple organizations with commas
            answer = ", ".join(orgs)
            
            # Create question with appropriate template based on number of organizations
            has_multiple = len(orgs) > 1
            question = f"In what organizations does {person_name} work?" if has_multiple else f"In what organization does {person_name} work?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_org_representative_questions(self, graph, doc_num):
        """Generate questions about organization representatives"""
        questions = []
        
        # SPARQL query to get all organizations and their employees
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?person; separator="|") as ?persons)
        WHERE {
            ?org a <http://example.org/base/Organization> .
            ?person <http://example.org/relation/isEmployedBy> ?org .
        }
        GROUP BY ?org
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            person_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            persons = [self.clean_uri(person) for person in person_uris]
            
            # Join multiple persons with commas
            answer = ", ".join(persons)
            
            # Create question with appropriate template based on number of representatives
            has_multiple = len(persons) > 1
            question = f"Who are the representatives of {org_name}?" if has_multiple else f"Who is the representative of {org_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_org_location_questions(self, graph, doc_num):
        """Generate questions about organization locations"""
        questions = []
        
        # SPARQL query to get all organizations and their locations
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?loc; separator="|") as ?locs)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?org
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            loc_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            locations = [self.clean_uri(loc) for loc in loc_uris]
            
            # Join multiple locations with commas
            answer = ", ".join(locations)
            
            # Create question with appropriate template based on number of locations
            has_multiple = len(locations) > 1
            question = f"What are the locations of {org_name}?" if has_multiple else f"What is the location of {org_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_location_company_questions(self, graph, doc_num):
        """Generate questions about companies at locations"""
        questions = []
        
        # SPARQL query to get all locations and their organizations
        query = """
        SELECT DISTINCT ?loc (GROUP_CONCAT(?org; separator="|") as ?orgs)
        WHERE {
            ?loc a <http://example.org/base/Location> .
            ?org <http://example.org/relation/hasLocationAt> ?loc .
        }
        GROUP BY ?loc
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row[0]
            org_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            loc_name = self.clean_uri(loc_uri)
            orgs = [self.clean_uri(org) for org in org_uris]
            
            # Join multiple organizations with commas
            answer = ", ".join(orgs)
            
            # Create question with appropriate template based on number of companies
            has_multiple = len(orgs) > 1
            question = f"Which companies are associated with {loc_name}?" if has_multiple else f"Which company is associated with {loc_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_location_type_questions(self, graph, doc_num):
        """Generate questions about location types"""
        questions = []
        
        # SPARQL query to get all locations and their types
        query = """
        SELECT DISTINCT ?loc (GROUP_CONCAT(?type; separator="|") as ?types)
        WHERE {
            ?loc a <http://example.org/base/Location> ;
                 <http://example.org/isInstanceOf/> ?type .
            FILTER(STRSTARTS(STR(?type), "http://example.org/location_type/"))
        }
        GROUP BY ?loc
        """
        
        results = graph.query(query)
        
        for row in results:
            loc_uri = row[0]
            type_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            loc_name = self.clean_uri(loc_uri)
            types = [self.clean_uri(type_uri) for type_uri in type_uris]
            
            # Join multiple types with commas
            answer = ", ".join(types)
            
            # Create question with appropriate template based on number of types
            has_multiple = len(types) > 1
            question = f"What are the types of location {loc_name}?" if has_multiple else f"What type of location is {loc_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

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

    def generate_org_role_questions(self, graph, doc_num):
        """Generate questions about organization roles"""
        questions = []
        
        # Get role-subrole mapping
        role_subrole_map = self.get_org_role_subrole_map(graph)
        
        # SPARQL query to get organizations and their roles/sub-roles
        query = """
        SELECT DISTINCT ?org (GROUP_CONCAT(?role; separator="|") as ?roles)
        WHERE {
            ?org a <http://example.org/base/Organization> ;
                <http://example.org/isInstanceOf/> ?role .
            {
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
            } UNION {
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_sub_role/"))
            }
        }
        GROUP BY ?org
        """
        
        results = graph.query(query)
        
        for row in results:
            org_uri = row[0]
            role_uris = row[1].split('|')
            
            # Clean the URIs to get readable text
            org_name = self.clean_uri(org_uri)
            roles = [self.clean_uri(role) for role in role_uris]
            
            # Separate roles and sub-roles
            main_roles = set()
            sub_roles = set()
            for role in roles:
                if any(role in subroles for subroles in role_subrole_map.values()):
                    sub_roles.add(role)
                else:
                    main_roles.add(role)
            
            # Create answer based on roles and sub-roles
            answers = []
            for main_role in main_roles:
                # Check if this role has any sub-roles in our data
                matching_sub_roles = sub_roles.intersection(role_subrole_map.get(main_role, set()))
                if matching_sub_roles:
                    # For each sub-role of this main role, create "sub_role role" pair
                    for sub_role in matching_sub_roles:
                        answers.append(f"{sub_role} {main_role}")
                else:
                    # If no sub-roles, just add the main role
                    answers.append(main_role)
            
            # Skip if no answers found
            if not answers:
                continue
                
            # Join all answers with commas
            answer = ", ".join(sorted(answers))
            
            # Create question with appropriate template based on number of roles
            has_multiple = len(answers) > 1
            question = f"What are the roles of {org_name} in the agreement?" if has_multiple else f"What is the role of {org_name} in the agreement?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num,
                'multiple_answers': 1 if has_multiple else 0
            })
        
        return questions

    def generate_company_by_role_questions(self, graph, doc_num):
        """Generate questions about companies with specific roles"""
        questions = []
        
        # Get role-subrole mapping
        role_subrole_map = self.get_org_role_subrole_map(graph)
        
        # SPARQL query to get organizations and their roles/sub-roles
        query = """
        SELECT DISTINCT ?role ?sub_role (GROUP_CONCAT(?org; separator="|") as ?orgs)
        WHERE {
            ?org a <http://example.org/base/Organization> .
            {
                ?org <http://example.org/isInstanceOf/> ?role .
                FILTER(STRSTARTS(STR(?role), "http://example.org/org_role/"))
                OPTIONAL {
                    ?org <http://example.org/isInstanceOf/> ?sub_role .
                    ?sub_role rdfs:subClassOf ?role .
                    FILTER(STRSTARTS(STR(?sub_role), "http://example.org/org_sub_role/"))
                }
            }
        }
        GROUP BY ?role ?sub_role
        """
        
        results = graph.query(query)
        
        role_org_map = {}  # To store role -> orgs mapping
        
        for row in results:
            role_uri = row[0]
            sub_role_uri = row[1] if row[1] else None
            org_uris = row[2].split('|')
            
            # Clean the URIs to get readable text
            role = self.clean_uri(role_uri)
            sub_role = self.clean_uri(sub_role_uri) if sub_role_uri else None
            orgs = [self.clean_uri(org) for org in org_uris]
            
            # Create key for role combination
            role_key = f"{sub_role} {role}" if sub_role else role
            
            # Store organizations for this role combination
            if role_key not in role_org_map:
                role_org_map[role_key] = set()
            role_org_map[role_key].update(orgs)
        
        # Generate questions for each role combination
        for role_key, orgs in role_org_map.items():
            # Skip if no organizations found
            if not orgs:
                continue
                
            # Sort and join organization names
            answer = ", ".join(sorted(orgs))
            
            # Create question with appropriate template based on number of organizations
            has_multiple = len(orgs) > 1
            if " " in role_key:  # Has both sub-role and role
                question = f"What companies are the {role_key} in the agreement?" if has_multiple else f"What company is the {role_key} in the agreement?"
            else:  # Only has main role
                question = f"What companies are the {role_key} in the agreement?" if has_multiple else f"What company is the {role_key} in the agreement?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
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
                position_questions = self.generate_position_questions(graph, doc_num)
                org_questions = self.generate_person_org_questions(graph, doc_num)
                representative_questions = self.generate_org_representative_questions(graph, doc_num)
                org_location_questions = self.generate_org_location_questions(graph, doc_num)
                location_company_questions = self.generate_location_company_questions(graph, doc_num)
                location_type_questions = self.generate_location_type_questions(graph, doc_num)
                org_role_questions = self.generate_org_role_questions(graph, doc_num)
                company_role_questions = self.generate_company_by_role_questions(graph, doc_num)
                
                # Combine all questions
                all_questions.extend(position_questions)
                all_questions.extend(org_questions)
                all_questions.extend(representative_questions)
                all_questions.extend(org_location_questions)
                all_questions.extend(location_company_questions)
                all_questions.extend(location_type_questions)
                all_questions.extend(org_role_questions)
                all_questions.extend(company_role_questions)
                
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
    df.to_csv('qa_dataframe.csv', index=False)
    print(f"Generated {len(df)} questions and saved to qa_dataframe.csv")
    
    # Display sample questions of each type
    print("\nSample questions:")
    print("\nPosition questions:")
    print(df[df['question'].str.startswith('What is the position')].head(2))
    print("\nOrganization questions:")
    print(df[df['question'].str.startswith('In what organization')].head(2))
    print("\nRepresentative questions:")
    print(df[df['question'].str.startswith('Who is the representative')].head(2))
    print("\nLocation questions:")
    print(df[df['question'].str.startswith('What is the location')].head(2))
    print("\nCompany at location questions:")
    print(df[df['question'].str.startswith('Which company')].head(2))
    print("\nLocation type questions:")
    print(df[df['question'].str.startswith('What type of location')].head(2))
    print("\nOrganization role questions:")
    print(df[df['question'].str.startswith('What is the role')].head(2))
    print("\nCompany by role questions:")
    print(df[df['question'].str.contains('company is the|companies are the')].head(2))

if __name__ == "__main__":
    main() 