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
            
            # Create question and answer
            question = f"What is the position of {person_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
            
            # Create question and answer
            question = f"In what organization does {person_name} work?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
            
            # Create question and answer
            question = f"Who is the representative of {org_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
            
            # Create question and answer
            question = f"What is the location of {org_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
            
            # Create question and answer
            question = f"Which company is associated with {loc_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
            
            # Create question and answer
            question = f"What type of location is {loc_name}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,
                'num_set_operations': 0,
                'document_number': doc_num
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
                
                # Combine all questions
                all_questions.extend(position_questions)
                all_questions.extend(org_questions)
                all_questions.extend(representative_questions)
                all_questions.extend(org_location_questions)
                all_questions.extend(location_company_questions)
                all_questions.extend(location_type_questions)
                
            except Exception as e:
                print(f"Error processing {ttl_file}: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(all_questions)
        
        # Reorder columns to put document_number first
        columns_order = ['document_number', 'question', 'answer', 'num_hops', 'num_set_operations']
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

if __name__ == "__main__":
    main() 