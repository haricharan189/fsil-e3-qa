from rdflib import Graph
import os
import glob
import re  # Import the regular expressions module
from typing import List, Set

import os
import glob
import re
from rdflib import Graph
import os
import glob
import re
from rdflib import Graph

class QueryGenerator:
    def __init__(self, ontology_file_path, data_directory):
        self.ontology_graph = self.load_graph(ontology_file_path)
        self.data_files = glob.glob(os.path.join(data_directory, "*.ttl"))  # Get all .ttl files in the directory

    def load_graph(self, ttl_file_path):
        """Load a TTL file into an RDF graph."""
        graph = Graph()
        graph.parse(ttl_file_path, format="ttl")
        return graph

    def L1_person(self, data_graph, document_name):
        """Retrieve subclasses of person:Person, generate NLQs, and save them."""
        # Step 1: Retrieve subclasses of person:Person from the ontology
        query_subclasses = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX person: <http://example.org/person/>
        SELECT ?subclass WHERE {
            ?subclass rdfs:subClassOf person:Person .
        }
        """
        try:
            subclass_results = self.ontology_graph.query(query_subclasses)
            subclasses = {str(row.subclass).split("/")[-1] for row in subclass_results}
        except Exception as e:
            print(f"Error querying subclasses of person:Person: {e}")
            subclasses = set()

        # Debugging output
        

        # Step 2: Query the data graph for Person instances
        query_nlq = """
        PREFIX person: <http://example.org/person/>
        PREFIX rel:   <http://example.org/relation/>
        SELECT ?name ?position ?organization WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?organization .
            BIND(REPLACE(STR(?person), ".*/", "") AS ?name)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        try:
            results = data_graph.query(query_nlq)
        except Exception as e:
            print(f"Error executing query on data graph for document {document_name}: {e}")
            return []

        # Step 3: Generate NLQs using subclasses and data graph results
        nlqs = []
        for row in results:
            try:
                name, position, organization = row
                position_name = position.split("/")[-1]
                if position_name in subclasses:
                    nlqs.append(f"What is the position of {name} in {organization}?")
                    nlqs.append(f"Who is the {position_name.lower()} of {organization}?")
                    nlqs.append(f"Where does {name} work as {position_name.lower()}?")
            except Exception as e:
                print(f"Error processing row {row}: {e}")

        # Debugging output
        print(f"Generated person-related NLQs for {document_name}: {nlqs}")

        # Step 4: Clean and save the queries
        cleaned_queries = [self.clean_query(query) for query in nlqs]
        self.save_queries(cleaned_queries, document_name, append=True, prefix="L1")

        return nlqs  # Optionally return the NLQs if needed

    def Level_1_location(self, data_graph, document_name):
        """Generate natural language queries for the Location class and append to the file."""
        nlqs = []

        # Query the data graph for Location instances and their associated organization
        location_query = """
        PREFIX loc: <http://example.org/location/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?location ?organization WHERE {
            ?loc a loc:Location ;
                 rel:isLocationOf ?organization .
            BIND(REPLACE(STR(?loc), ".*/", "") AS ?location)
            BIND(REPLACE(STR(?organization), ".*/", "") AS ?organization)
        }
        """
        results = data_graph.query(location_query)
        
        # Generate the queries for each location and organization pair
        for row in results:
            location = row[0].split("/")[-1]  # Extract the location name
            organization = row[1].split("/")[-1].replace("%5Cn", "").replace("%2C", ",")  # Clean organization name
            
            # Generate the queries
            nlqs.append(f"Where is {organization} located?")
            nlqs.append(f"Which organization is located at {location}?")
        
        # Clean the queries before saving
        cleaned_queries = [self.clean_query(query) for query in set(nlqs)]

        # Save the generated queries to a text file for the document
        self.save_queries(cleaned_queries, document_name, append=True, prefix="L1")  # Specify prefix

    def Level_1_Roles(self, data_graph, document_name):
        """Retrieve roles and sub-roles from the ontology graph and generate NLQs for them."""
        # Retrieve roles and sub-roles from the ontology graph
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX org: <http://example.org/organization/>
        SELECT ?role ?subRole WHERE {
            ?role rdfs:subClassOf org:Organization .
            OPTIONAL { ?subRole rdfs:subClassOf ?role . }
        }
        """
        results = self.ontology_graph.query(query)
        roles_and_sub_roles = {}

        for row in results:
            role = str(row.role).split("/")[-1]
            sub_role = str(row.subRole).split("/")[-1] if row.subRole else None
            if role not in roles_and_sub_roles:
                roles_and_sub_roles[role] = set()
            if sub_role:
                roles_and_sub_roles[role].add(sub_role)

        nlqs = []

        # Query the data graph for Organization-related roles
        org_query = """
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?organization ?role ?employee WHERE {
            ?org a org:Organization ;
                 a ?role ;
                 rel:hasEmployee ?employee .
            FILTER (?role != org:Organization)  # Add filter to exclude main Organization class
            BIND(REPLACE(STR(?org), ".*/", "") AS ?organization)
            BIND(REPLACE(STR(?role), ".*/", "") AS ?role)
            BIND(REPLACE(STR(?employee), ".*/", "") AS ?employee)
        }
        """
        results = data_graph.query(org_query)

        for row in results:
            organization, role, employee = row
            role_name = role.split("/")[-1]
            sub_roles = roles_and_sub_roles.get(role_name, set())

            # Generate queries with the role and sub-role
            nlqs.append(f"What is the role of {organization}?")
            for sub_role in sub_roles:
                nlqs.append(f"Who is the {sub_role.lower()} {role_name.lower()} in the agreement?")

        # Clean the queries before saving
        cleaned_queries = [self.clean_query(query) for query in set(nlqs)]

        # Save the generated queries to a text file for the document
        self.save_queries(cleaned_queries, document_name, append=True, prefix="L1")  # Specify prefix

    def L2_person_org(self, data_graph, document_name):
        """Retrieve combined roles and generate Level 2 NLQs based on Person-Organization relationships."""
        # Retrieve combined roles from the ontology graph
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX org: <http://example.org/organization/>
        SELECT ?role ?subRole WHERE {
            ?role rdfs:subClassOf org:Organization .
            OPTIONAL { ?subRole rdfs:subClassOf ?role . }
        }
        """
        results = self.ontology_graph.query(query)
        combined_roles = {}

        for row in results:
            role = str(row.role).split("/")[-1]
            sub_role = str(row.subRole).split("/")[-1] if row.subRole else None
            combined_roles[role] = sub_role

        nlqs = []

        # SPARQL Query for Person–Organization relationships
        person_org_query = """
        PREFIX person: <http://example.org/person/>
        PREFIX org: <http://example.org/organization/>
        PREFIX rel: <http://example.org/relation/>
        SELECT ?person ?position ?organization ?role WHERE {
            ?person a person:Person ;
                    a ?position ;
                    rel:isEmployedBy ?org .
            ?org a org:Organization ;
                 a ?role .
            FILTER(?position != person:Person && ?role != org:Organization)
        }
        """
        results = data_graph.query(person_org_query)
        for row in results:
            person, position, organization, role = row
            person_name = str(person).split("/")[-1]
            position_name = str(position).split("/")[-1]
            organization_name = str(organization).split("/")[-1]
            role_name = str(role).split("/")[-1]

            # Combine sub-role and role
            sub_role = combined_roles.get(role_name, None)
            role_with_sub_role = f"{sub_role} {role_name}" if sub_role else role_name

            # Generate NLQs for Person–Organization
            nlqs.append(f"Who is the {position_name.lower()} of the organization that is the {role_with_sub_role.lower()}?")
            nlqs.append(f"{person_name} holds what position in the organization that is the {role_with_sub_role.lower()}?")
            nlqs.append(f"What is the role of the organization in which {person_name} is {position_name.lower()}?")

        # Clean the queries before saving
        cleaned_queries = [self.clean_query(query) for query in nlqs]

        # Save the generated queries to a text file for the document
        self.save_queries(cleaned_queries, document_name, append=True, prefix="L2")  # Specify prefix

    def save_queries(self, queries, document_name, append=False, prefix="L1"):
        """Save the generated queries to a text file in the queries directory."""
        queries_dir = os.path.join(os.getcwd(), "queries")
        os.makedirs(queries_dir, exist_ok=True)  # Create the queries directory if it doesn't exist
        file_path = os.path.join(queries_dir, f"{prefix}_queries_{document_name}.txt")  # Use prefix for naming
        
        mode = 'a' if append else 'w'  # Append mode if specified, otherwise write mode
        with open(file_path, mode) as f:
            for query in queries:
                f.write(query + "\n")

    def clean_query(self, query):
        """Remove unwanted special characters from the query while preserving spaces."""
        # Remove unwanted patterns like 5Cn, C2, A0, etc. (case insensitive)
        cleaned_query = re.sub(r'[\d]+Cn|[\d]+C|C[0-9]+|A[0-9]+|5Cn|C2|A0+26', '', query, flags=re.IGNORECASE)  # Remove specific patterns
        cleaned_query = re.sub(r'[%\\]+', ' ', cleaned_query)  # Replace % and \ with a space
        cleaned_query = re.sub(r'[_]+', ' ', cleaned_query)  # Replace underscores with a space
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query)  # Normalize multiple spaces to a single space
        return cleaned_query.strip()  # Remove leading/trailing spaces

if __name__ == "__main__":
    # Load the ontology TTL file
    ontology_file_path = "./extracted_content/ontology.ttl"
    data_directory = "./extracted_content"  # Directory containing .ttl files

    # Create an instance of QueryGenerator
    query_generator = QueryGenerator(ontology_file_path, data_directory)

    # Iterate through all data files and generate NLQs
    for data_file_path in query_generator.data_files:
        data_graph = query_generator.load_graph(data_file_path)
        document_name = os.path.basename(data_file_path).replace('.ttl', '')
        
        # Generate person queries
        nlqs_person = query_generator.L1_person(data_graph, document_name)  # Call the new combined method
        
        # Generate location queries
        nlqs_location = query_generator.Level_1_location(data_graph, document_name)  
        
        # Generate role queries
        nlqs_roles = query_generator.Level_1_Roles(data_graph, document_name)  # Call the new roles method
        
        # Generate Level 2 queries
        nlqs_level2 = query_generator.L2_person_org(data_graph, document_name)



     

      
     