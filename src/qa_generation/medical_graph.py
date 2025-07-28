import json
import re

from pathlib import Path


class MedicalGraph:
    def __init__(self):
        self.pathogens: set[str] = set()
        self.vectors: set[str] = set()
        self.diseases: set[str] = set()
        self.medications: set[str] = set()

        self.pathogen_vector: dict[set] = {}
        self.pathogen_disease: dict[set] = {}
        self.disease_medication: dict[set] = {}

        # reverse relations, double-memory but QA-generation effective
        self.vector_pathogen: dict[set] = {}
        self.disease_pathogen: dict[set] = {}
        self.medication_disease: dict[set] = {}

    def _add_entity(self, entity: str, entity_type: str):
        if entity_type == "Pathogen":
            self.pathogens.add(entity)
            self.pathogen_vector[entity] = set()
            self.pathogen_disease[entity] = set()
        elif entity_type == "Vector":
            self.vectors.add(entity)
            self.vector_pathogen[entity] = set()
        elif entity_type == "Disease":
            self.diseases.add(entity)
            self.disease_medication[entity] = set()
            self.disease_pathogen[entity] = set()
        elif entity_type == "Medication":
            self.medications.add(entity)
            self.medication_disease[entity] = set()
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def _add_relation(self, source: str, target: str, relation_type: str):
        if relation_type == "Pathogen_Vector":
            self.pathogen_vector[source].add(target)
            self.vector_pathogen[target].add(source)
        elif relation_type == "Pathogen_Disease":
            self.pathogen_disease[source].add(target)
            self.disease_pathogen[target].add(source)
        elif relation_type == "Disease_Medication":
            self.disease_medication[source].add(target)
            self.medication_disease[target].add(source)
        else:
            raise ValueError(f"Unknown relation type: {relation_type}")

    @staticmethod
    def __clean_entity_text(s: str):
        # Keep only Latin letters, spaces and "-"
        s = re.sub(r'[^a-zA-Z\s\-]', '', s)
        # Collapse multiple spaces into one
        s = re.sub(r'\s+', ' ', s)
        # Strip and convert to lowercase
        return s.strip().lower()

    def _add_from_annotation(self, annotation: dict, id_to_entity: dict):
        if 'value' in annotation:
            # annotation of entity
            entity = MedicalGraph.__clean_entity_text(
                annotation['value']['text'])
            entity_type = annotation['value']['hypertextlabels'][0]
            entity_id = annotation['id']
            self._add_entity(entity, entity_type)
            id_to_entity[entity_id] = (entity, entity_type)
        else:
            # annotation of relation
            from_entity = id_to_entity[annotation['from_id']]
            to_entity = id_to_entity[annotation['to_id']]
            source = from_entity[0]
            target = to_entity[0]
            relation_type = "{}_{}".format(from_entity[1], to_entity[1])
            self._add_relation(source, target, relation_type)

    def add_graph_from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        id_to_entity = {}
        for i in range(len(data)):
            for j in range(len(data[i]['annotations'])):
                annotations = data[i]['annotations'][j]['result']
                for annotation in annotations:
                    self._add_from_annotation(annotation, id_to_entity)

    def add_graph_from_directory(self, dir_path: str):
        dir_path = Path(dir_path)
        json_files = list(dir_path.rglob("*.json"))
        for file_path in json_files:
            self.add_graph_from_file(file_path)

    def serialize(self, file_path: str) -> dict:
        _pathogen_vector = {key: list(value) for key, value
                            in self.pathogen_vector.items()}
        _pathogen_disease = {key: list(value) for key, value
                             in self.pathogen_disease.items()}
        _disease_medication = {key: list(value) for key, value
                               in self.disease_medication.items()}
        _vector_pathogen = {key: list(value) for key, value
                            in self.vector_pathogen.items()}
        _disease_pathogen = {key: list(value) for key, value
                             in self.disease_pathogen.items()}
        _medication_disease = {key: list(value) for key, value
                               in self.medication_disease.items()}
        data = {
            "pathogens": list(self.pathogens),
            "vectors": list(self.vectors),
            "diseases": list(self.diseases),
            "medications": list(self.medications),
            "pathogen_vector": _pathogen_vector,
            "pathogen_disease": _pathogen_disease,
            "disease_medication": _disease_medication,
            "vector_pathogen": _vector_pathogen,
            "disease_pathogen": _disease_pathogen,
            "medication_disease": _medication_disease
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
        return data

    @classmethod
    def deserialize(cls, file_path: str) -> 'MedicalGraph':
        with open(file_path, 'r') as f:
            data = json.load(f)
        graph = cls()
        graph.pathogens = set(data.get("pathogens", []))
        graph.vectors = set(data.get("vectors", []))
        graph.diseases = set(data.get("diseases", []))
        graph.medications = set(data.get("medications", []))
        graph.pathogen_vector = {key: set(value) for key, value
                                 in data.get("pathogen_vector", {}).items()}
        graph.pathogen_disease = {key: set(value) for key, value
                                  in data.get("pathogen_disease", {}).items()}
        graph.disease_medication = {key: set(value) for key, value
                                    in data.get("disease_medication", {}).items()}
        graph.vector_pathogen = {key: set(value) for key, value
                                 in data.get("vector_pathogen", {}).items()}
        graph.disease_pathogen = {key: set(value) for key, value
                                  in data.get("disease_pathogen", {}).items()}
        graph.medication_disease = {key: set(value) for key, value
                                    in data.get("medication_disease", {}).items()}
        return graph

    def __str__(self):
        return json.dumps(self.serialize(), indent=4)
