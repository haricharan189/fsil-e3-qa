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
            
            # Find positions held by person1 but not by person2
            unique_positions = positions1 - positions2
            
            # Create answer string
            if not unique_positions:
                answer = "None"
            else:
                answer = ", ".join(sorted(unique_positions))
            
            # Create question
            question = f"What is the position held by {person1} but not by {person2}?"
            
            questions.append({
                'question': question,
                'answer': answer,
                'num_hops': 1,  # Need to check positions of both people
                'num_set_operations': 2,  # One set difference operation
                'document_number': doc_num
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
                        # Create question
                        question = f"Who is the {pos1} and {pos2} of {org_name}?"
                        
                        questions.append({
                            'question': question,
                            'answer': person_name,
                            'num_hops': 2,  # Need to check both positions
                            'num_set_operations': 1,  # Need to find intersection of people with both positions
                            'document_number': doc_num
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
                        question = f"Who is the {position} of the company associated with {location}?"
                        
                        if question not in position_holders:
                            position_holders[question] = set()
                        position_holders[question].add(person_name)
            
            # Create questions with combined answers
            for question, people in position_holders.items():
                # Sort names alphabetically and join with commas
                answer = ", ".join(sorted(people))
                
                questions[question] = {
                    'question': question,
                    'answer': answer,
                    'num_hops': 3,  # Location -> Organization -> Person -> Position
                    'num_set_operations': 0,
                    'document_number': doc_num
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
                # For each position in the organization
                for position, holders in info['positions'].items():
                    # Create question
                    question = f"Who is the {position} of the company associated where {employee} is employed?"
                    
                    # Sort and join position holders
                    answer = ", ".join(sorted(holders))
                    
                    questions[question] = {
                        'question': question,
                        'answer': answer,
                        'num_hops': 3,  # Person -> Organization -> Other Person -> Position
                        'num_set_operations': 0,
                        'document_number': doc_num
                    }
        
        return list(questions.values())
    
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
                
                # Combine all questions
                all_questions.extend(position_diff_questions)
                all_questions.extend(multi_position_questions)
                all_questions.extend(location_position_questions)
                all_questions.extend(employee_org_position_questions)
                
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
