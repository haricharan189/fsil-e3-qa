"""
ground_truth.py

Changes:
 - Paths from config
 - Writes L1_answers_..., L2_answers_... to ground_truth/
 - Additional comments for clarity
"""

from rdflib import Graph
import os
import glob
import re

import config

class GroundTruthExtractor:
    def __init__(self, ontology_file_path, data_directory):
        self.ontology_graph = self.load_graph(ontology_file_path)
        self.data_files     = glob.glob(os.path.join(data_directory, "*.ttl"))

    def load_graph(self, ttl_file_path):
        g = Graph()
        g.parse(ttl_file_path, format="ttl")
        return g

    def Level_1_person_answers(self, data_graph, document_name):
        """
        Extract ground truth answers for Person
        """
        answers = []
        query = """
        PREFIX person: <http://example.org/person/>
        PREFIX rel: <http://example.org/relation/>
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
            org_name      = str(organization).split("/")[-1]
            answers.append(f"Q: What is the position of {name} in {org_name}?")
            answers.append(f"A: {position_name}")
            answers.append(f"Q: Who is the {position_name.lower()} of {org_name}?")
            answers.append(f"A: {name}")
            answers.append(f"Q: Where does {name} work as {position_name.lower()}?")
            answers.append(f"A: {org_name}")

        self.save_answers(answers, document_name)
        return answers

    def Level_1_location_answers(self, data_graph, document_name):
        """
        Extract ground truth answers for Location
        """
        answers = []
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
            loc_name = str(location).split("/")[-1]
            org_name = str(organization).split("/")[-1]
            answers.append(f"Q: Where is {org_name} located?")
            answers.append(f"A: {loc_name}")
            answers.append(f"Q: Which organization is located at {loc_name}?")
            answers.append(f"A: {org_name}")

        self.save_answers(answers, document_name, append=True)
        return answers

    def Level_1_Roles_answers(self, data_graph, document_name):
        """
        Extract ground truth answers for Organization roles
        """
        answers = []
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
            org_name  = str(organization).split("/")[-1]
            answers.append(f"Q: What is the role of {org_name}?")
            answers.append(f"A: {role_name}")

        self.save_answers(answers, document_name, append=True)
        return answers

    def L2_person_org_answers(self, data_graph, document_name):
        """
        Extract ground truth answers for Person-Organization relationship
        """
        answers = []
        query = """
        PREFIX person: <http://example.org/person/>
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
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
            person_name     = str(person).split("/")[-1]
            position_name   = str(position).split("/")[-1]
            org_name        = str(organization).split("/")[-1]
            role_name       = str(role).split("/")[-1]
            answers.append(f"Q: Who is the {position_name.lower()} of the organization that is the {role_name.lower()}?")
            answers.append(f"A: {person_name}")
            answers.append(f"Q: {person_name} holds what position in the organization that is the {role_name.lower()}?")
            answers.append(f"A: {position_name}")
            answers.append(f"Q: What is the role of the organization in which {person_name} is {position_name.lower()}?")
            answers.append(f"A: {role_name}")

        self.save_answers(answers, document_name, level=2)
        return answers

    def save_answers(self, answers, document_name, append=False, level=1):
        """
        Writes Q/A pairs to L{level}_answers_{document_name}.txt
        """
        os.makedirs(config.GROUND_TRUTH_DIR, exist_ok=True)
        file_path = os.path.join(config.GROUND_TRUTH_DIR, f"L{level}_answers_{document_name}.txt")
        mode = 'a' if append else 'w'
        
        filtered_answers = []
        seen_questions = set()
        i = 0
        while i < len(answers):
            question = answers[i]
            answer   = answers[i+1] if i+1 < len(answers) else None
            question_text = question.lower().strip()
            last_word     = question_text.split()[-1].rstrip('?') if question_text else ""

            if (answer 
                and not answer.strip().endswith("Person") 
                and not answer.strip().endswith("Organization")
                and last_word not in ["person", "organization"]
                and "person" not in question_text
                and question not in seen_questions):
                
                filtered_answers.extend([question, answer])
                seen_questions.add(question)
            i += 2

        with open(file_path, mode) as f:
            for ans in filtered_answers:
                f.write(self.clean_text(ans) + "\n")

    def clean_text(self, text):
        cleaned_text = re.sub(r'[\d]+Cn|[\d]+C|C[0-9]+|A[0-9]+|5Cn|C2|A0+26', '', text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'[%\\]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[_]+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()

def main():
    """
    If you want to run ground truth extraction standalone.
    """
    gte = GroundTruthExtractor(
        ontology_file_path=os.path.join(config.EXTRACTED_CONTENT_DIR, "ontology.ttl"),
        data_directory=config.EXTRACTED_CONTENT_DIR
    )
    for data_file_path in gte.data_files:
        data_graph = gte.load_graph(data_file_path)
        doc_name   = os.path.basename(data_file_path).replace('.ttl', '')
        gte.Level_1_person_answers(data_graph, doc_name)
        gte.Level_1_location_answers(data_graph, doc_name)
        gte.Level_1_Roles_answers(data_graph, doc_name)
        gte.L2_person_org_answers(data_graph, doc_name)

if __name__ == "__main__":
    main()
