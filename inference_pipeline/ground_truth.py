from rdflib import Graph
import os
import glob
import re

class GroundTruthExtractor:
    def __init__(self, ontology_file_path, data_directory):
        self.ontology_graph = self.load_graph(ontology_file_path)
        self.data_files = glob.glob(os.path.join(data_directory, "*.ttl"))

    def load_graph(self, ttl_file_path):
        """Load a TTL file into an RDF graph."""
        graph = Graph()
        graph.parse(ttl_file_path, format="ttl")
        return graph

    def Level_1_person_answers(self, data_graph, document_name):
        """Extract ground truth answers for Level 1 person-related queries."""
        answers = []
        
        query = """
        PREFIX person: <http://example.org/person/>
        PREFIX rel: <http://example.org/relation/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?name ?position ?organization WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?organization .
            FILTER(?position != person:Person)
            BIND(REPLACE(STR(?person), ".*/", "") AS ?name)
            BIND(REPLACE(STR(?position), ".*/", "") AS ?position)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        results = data_graph.query(query)
        
        for row in results:
            name, position, organization = row
            position_name = str(position).split("/")[-1]
            org_name = str(organization).split("/")[-1]
            
            # Store answers for different question patterns
            answers.append(f"Q: What is the position of {name} in {org_name}?")
            answers.append(f"A: {position_name}")
            
            answers.append(f"Q: Who is the {position_name.lower()} of {org_name}?")
            answers.append(f"A: {name}")
            
            answers.append(f"Q: Where does {name} work as {position_name.lower()}?")
            answers.append(f"A: {org_name}")

        # Save the answers
        self.save_answers(answers, document_name)
        return answers

    def Level_1_location_answers(self, data_graph, document_name):
        """Extract ground truth answers for Level 1 location-related queries."""
        answers = []
        
        # Query for locations and organizations
        query = """
        PREFIX loc: <http://example.org/location/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?location ?organization WHERE {
            ?loc a loc:Location ;
                 rel:isLocationOf ?organization .
            BIND(REPLACE(STR(?loc), ".*/", "") AS ?location)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        results = data_graph.query(query)
        
        for row in results:
            location, organization = row
            location_name = str(location).split("/")[-1]
            org_name = str(organization).split("/")[-1]
            
            answers.append(f"Q: Where is {org_name} located?")
            answers.append(f"A: {location_name}")
            
            answers.append(f"Q: Which organization is located at {location_name}?")
            answers.append(f"A: {org_name}")

        # Save the answers
        self.save_answers(answers, document_name, append=True)
        return answers

    def Level_1_Roles_answers(self, data_graph, document_name):
        """Extract ground truth answers for Level 1 role-related queries."""
        answers = []
        
        # Query for organizations and their roles
        query = """
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?organization ?role ?employee WHERE {
            ?org a org:Organization ;
                 a ?role ;
                 rel:hasEmployee ?employee .
            FILTER (?role != org:Organization)
            BIND(REPLACE(STR(?org), ".*/", "") AS ?organization)
            BIND(REPLACE(STR(?role), ".*/", "") AS ?role)
            BIND(REPLACE(STR(?employee), ".*/", "") AS ?employee)
        }
        """
        results = data_graph.query(query)
        
        for row in results:
            organization, role, employee = row
            role_name = str(role).split("/")[-1]
            org_name = str(organization).split("/")[-1]
            
            answers.append(f"Q: What is the role of {org_name}?")
            answers.append(f"A: {role_name}")

        # Save the answers
        self.save_answers(answers, document_name, append=True)
        return answers

    def L2_person_org_answers(self, data_graph, document_name):
        """Extract ground truth answers for Level 2 person-organization relationship queries."""
        answers = []
        
        query = """
        PREFIX person: <http://example.org/person/>
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?person ?position ?organization ?role WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?org .
            ?org a org:Organization ;
                 a ?role .
            FILTER(?position != person:Person && ?role != org:Organization)
            BIND(REPLACE(STR(?person), ".*/", "") AS ?person)
            BIND(REPLACE(STR(?position), ".*/", "") AS ?position)
            BIND(REPLACE(STR(?org), ".*/", "") AS ?organization)
            BIND(REPLACE(STR(?role), ".*/", "") AS ?role)
        }
        """
        results = data_graph.query(query)
        
        for row in results:
            person, position, organization, role = row
            person_name = str(person).split("/")[-1]
            position_name = str(position).split("/")[-1]
            org_name = str(organization).split("/")[-1]
            role_name = str(role).split("/")[-1]
            
            answers.append(f"Q: Who is the {position_name.lower()} of the organization that is the {role_name.lower()}?")
            answers.append(f"A: {person_name}")
            
            answers.append(f"Q: {person_name} holds what position in the organization that is the {role_name.lower()}?")
            answers.append(f"A: {position_name}")
            
            answers.append(f"Q: What is the role of the organization in which {person_name} is {position_name.lower()}?")
            answers.append(f"A: {role_name}")

        # Save the Level 2 answers
        self.save_answers(answers, document_name, level=2)
        return answers

    def save_answers(self, answers, document_name, append=False, level=1):
        """Save the extracted answers to a text file in the ground_truth directory."""
        ground_truth_dir = os.path.join(os.getcwd(), "ground_truth")
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        file_path = os.path.join(ground_truth_dir, f"L{level}_answers_{document_name}.txt")
        mode = 'a' if append else 'w'
        
        # Filter out Q&A pairs where answer is "Person" or "Organization" and handle duplicates
        filtered_answers = []
        seen_questions = set()
        i = 0
        while i < len(answers):
            question = answers[i]
            answer = answers[i + 1] if i + 1 < len(answers) else None
            
            # Extract the word before the question mark (if it exists)
            question_text = question.lower().strip()
            last_word = question_text.split()[-1].rstrip('?') if question_text else ""
            
            # Check if it's a valid Q&A pair and not a duplicate question
            if (answer and 
                not answer.strip().endswith("Person") and 
                not answer.strip().endswith("Organization") and
                last_word not in ["person", "organization"] and
                "person" not in question_text and  # Check for "person" anywhere in question
                question not in seen_questions):
                filtered_answers.extend([question, answer])
                seen_questions.add(question)
            i += 2
        
        with open(file_path, mode) as f:
            for answer in filtered_answers:
                cleaned_answer = self.clean_text(answer)
                f.write(cleaned_answer + "\n")

    def clean_text(self, text):
        """Clean the text by removing unwanted characters and normalizing spaces."""
        cleaned_text = re.sub(r'[\d]+Cn|[\d]+C|C[0-9]+|A[0-9]+|5Cn|C2|A0+26', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'[%\\]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[_]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

if __name__ == "__main__":
    # Initialize paths
    ontology_file_path = "./extracted_content/ontology.ttl"
    data_directory = "./extracted_content"

    # Create an instance of GroundTruthExtractor
    extractor = GroundTruthExtractor(ontology_file_path, data_directory)

    # Process each data file
    for data_file_path in extractor.data_files:
        data_graph = extractor.load_graph(data_file_path)
        document_name = os.path.basename(data_file_path).replace('.ttl', '')
        
        # Extract Level 1 answers
        extractor.Level_1_person_answers(data_graph, document_name)
        extractor.Level_1_location_answers(data_graph, document_name)
        extractor.Level_1_Roles_answers(data_graph, document_name)
        
        # Extract Level 2 answers
        extractor.L2_person_org_answers(data_graph, document_name)
