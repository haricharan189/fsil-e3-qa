from .medical_graph import MedicalGraph


class QATemplate:
    singular_template: str = None
    plural_template: str = None

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        raise NotImplementedError("Must be implemented in subclass.")
