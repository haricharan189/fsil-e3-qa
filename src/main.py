import csv
import inspect
import os

import qa_generation.templates as templates

from qa_generation import MedicalGraph, QATemplate

DATA_DIR: str = './../data/'
ANNOTATED_DATA_DIR: str = os.path.join(DATA_DIR, 'annotated')
GRAPHS_DIR: str = os.path.join(DATA_DIR, 'graphs')
os.makedirs(GRAPHS_DIR, exist_ok=True)
QA_DIR: str = os.path.join(DATA_DIR, 'qa')
os.makedirs(QA_DIR, exist_ok=True)


def run_all_templates(graph):
    for name, cls in inspect.getmembers(templates, inspect.isclass):
        if issubclass(cls, QATemplate) and cls is not QATemplate:
            instance = cls()
            qa_pairs = instance.generate(graph)
            for plurality in ['0', '1']:
                with open(os.path.join(QA_DIR,
                                       f"{str(instance).format(plurality)}.csv"),
                          'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(('question', 'answer'))
                    writer.writerows(qa_pairs[plurality])


if __name__ == '__main__':
    graph = MedicalGraph()
    graph.add_graph_from_directory(ANNOTATED_DATA_DIR)
    graph.serialize(os.path.join(GRAPHS_DIR, 'medical_graph.json'))
    all_qas = run_all_templates(graph)
