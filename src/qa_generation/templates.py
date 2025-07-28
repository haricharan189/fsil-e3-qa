from .medical_graph import MedicalGraph
from .qa_template import QATemplate


# 1-hop templates
# -------------------------------------

class PathogenByVectorTemplate(QATemplate):
    singular_template: str = "What pathogen is transmitted by {}?"
    plural_template: str = "What pathogens are transmitted by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vector in graph.vectors:
            pathogens = graph.vector_pathogen[vector]
            if len(pathogens) > 1:
                result['1'].append(
                    (self.plural_template.format(vector),
                     ",".join(sorted(pathogens)))
                )
            elif len(pathogens) == 1:
                result['0'].append(
                    (self.singular_template.format(vector),
                     next(iter(pathogens)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_pathogen_by_vector"


class VectorByPathogenTemplate(QATemplate):
    singular_template: str = "What vector transmits {}?"
    plural_template: str = "What vectors transmit {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for pathogen in graph.pathogens:
            vectors = graph.pathogen_vector[pathogen]
            if len(vectors) > 1:
                result['1'].append(
                    (self.plural_template.format(pathogen),
                     ",".join(sorted(vectors)))
                )
            elif len(vectors) == 1:
                result['0'].append(
                    (self.singular_template.format(pathogen),
                     next(iter(vectors)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_vector_by_pathogen"


class PathogenByDiseaseTemplate(QATemplate):
    singular_template: str = "What pathogen causes {}?"
    plural_template: str = "What pathogens cause {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for disease in graph.diseases:
            pathogens = graph.disease_pathogen[disease]
            if len(pathogens) > 1:
                result['1'].append(
                    (self.plural_template.format(disease),
                     ",".join(sorted(pathogens)))
                )
            elif len(pathogens) == 1:
                result['0'].append(
                    (self.singular_template.format(disease),
                     next(iter(pathogens)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_pathogen_by_disease"


class DiseaseByPathogenTemplate(QATemplate):
    singular_template: str = "What disease is caused by {}?"
    plural_template: str = "What diseases are caused by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for pathogen in graph.pathogens:
            diseases = graph.pathogen_disease[pathogen]
            if len(diseases) > 1:
                result['1'].append(
                    (self.plural_template.format(pathogen),
                     ",".join(sorted(diseases)))
                )
            elif len(diseases) == 1:
                result['0'].append(
                    (self.singular_template.format(pathogen),
                     next(iter(diseases)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_disease_by_pathogen"


class DiseaseByMedicationTemplate(QATemplate):
    singular_template: str = "What disease is treated by {}?"
    plural_template: str = "What diseases are treated by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for medication in graph.medications:
            diseases = graph.medication_disease[medication]
            if len(diseases) > 1:
                result['1'].append(
                    (self.plural_template.format(medication),
                     ",".join(sorted(diseases)))
                )
            elif len(diseases) == 1:
                result['0'].append(
                    (self.singular_template.format(medication),
                     next(iter(diseases)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_disease_by_medication"


class MedicationByDiseaseTemplate(QATemplate):
    singular_template: str = "What medication treats {}?"
    plural_template: str = "What medications treat {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for disease in graph.diseases:
            medications = graph.disease_medication[disease]
            if len(medications) > 1:
                result['1'].append(
                    (self.plural_template.format(disease),
                     ",".join(sorted(medications)))
                )
            elif len(medications) == 1:
                result['0'].append(
                    (self.singular_template.format(disease),
                     next(iter(medications)))
                )
        return result

    def __str__(self):
        return "t_{}_1_0_medication_by_disease"


class PathogenByVec1AndVec2Template(QATemplate):
    singular_template: str = "What pathogen is transmitted by {} and {}?"
    plural_template: str = "What pathogens are transmitted by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                common_pathogens = pathogens1.intersection(pathogens2)
                if len(common_pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                         ",".join(sorted(common_pathogens)))
                    )
                elif len(common_pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                         next(iter(common_pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_pathogen_by_vec1_and_vec2"


class VectorByPath1AndPath2Template(QATemplate):
    singular_template: str = "What vector transmits {} and {}?"
    plural_template: str = "What vectors transmit {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            vectors1 = graph.pathogen_vector[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                vectors2 = graph.pathogen_vector[path2]
                common_vectors = vectors1.intersection(vectors2)
                if len(common_vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                         ",".join(sorted(common_vectors)))
                    )
                elif len(common_vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                         next(iter(common_vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_vector_by_path1_and_path2"


class PathogenByDis1AndDis2Template(QATemplate):
    singular_template: str = "What pathogen causes {} and {}?"
    plural_template: str = "What pathogens cause {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            pathogens1 = graph.disease_pathogen[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                pathogens2 = graph.disease_pathogen[dis2]
                common_pathogens = pathogens1.intersection(pathogens2)
                if len(common_pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                         ",".join(sorted(common_pathogens)))
                    )
                elif len(common_pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                         next(iter(common_pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_pathogen_by_dis1_and_dis2"


class DiseaseByPath1AndPath2Template(QATemplate):
    singular_template: str = "What disease is caused by {} and {}?"
    plural_template: str = "What diseases are caused by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            diseases1 = graph.pathogen_disease[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                diseases2 = graph.pathogen_disease[path2]
                common_diseases = diseases1.intersection(diseases2)
                if len(common_diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                         ",".join(sorted(common_diseases)))
                    )
                elif len(common_diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                         next(iter(common_diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_disease_by_path1_and_path2"


class DiseaseByMed1AndMed2Template(QATemplate):
    singular_template: str = "What disease is treated by {} and {}?"
    plural_template: str = "What diseases are treated by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases1 = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med1]
                common_diseases = diseases1.intersection(diseases2)
                if len(common_diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                         ",".join(sorted(common_diseases)))
                    )
                elif len(common_diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                         next(iter(common_diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_disease_by_med1_and_med2"


class MedicationByDis1AndDis2Template(QATemplate):
    singular_template: str = "What medication treats {} and {}?"
    plural_template: str = "What medications treat {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            medication1 = graph.disease_medication[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                medication2 = graph.disease_medication[dis2]
                common_medications = medication1.intersection(medication2)
                if len(common_medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                         ",".join(sorted(common_medications)))
                    )
                elif len(common_medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                         next(iter(common_medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_medication_by_dis1_and_dis2"


class PathogenByVec1NotVec2Template(QATemplate):
    singular_template: str = "What pathogen is transmitted by {} but not {}?"
    plural_template: str = "What pathogens are transmitted by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                dif_pathogens = pathogens1.difference(pathogens2)
                if len(dif_pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                         ",".join(sorted(dif_pathogens)))
                    )
                elif len(dif_pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                         next(iter(dif_pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_pathogen_by_vec1_not_vec2"


class VectorByPath1NotPath2Template(QATemplate):
    singular_template: str = "What vector transmits {} and {}?"
    plural_template: str = "What vectors transmit {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            vectors1 = graph.pathogen_vector[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                vectors2 = graph.pathogen_vector[path2]
                dif_vectors = vectors1.difference(vectors2)
                if len(dif_vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                         ",".join(sorted(dif_vectors)))
                    )
                elif len(dif_vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                         next(iter(dif_vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_vector_by_path1_not_path2"


class PathogenByDis1NotDis2Template(QATemplate):
    singular_template: str = "What pathogen causes {} but not {}?"
    plural_template: str = "What pathogens cause {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            pathogens1 = graph.disease_pathogen[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                pathogens2 = graph.disease_pathogen[dis2]
                dif_pathogens = pathogens1.difference(pathogens2)
                if len(dif_pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                         ",".join(sorted(dif_pathogens)))
                    )
                elif len(dif_pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                         next(iter(dif_pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_pathogen_by_dis1_not_dis2"


class DiseaseByPath1NotPath2Template(QATemplate):
    singular_template: str = "What disease is caused by {} but not {}?"
    plural_template: str = "What diseases are caused by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            diseases1 = graph.pathogen_disease[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                diseases2 = graph.pathogen_disease[path2]
                dif_diseases = diseases1.difference(diseases2)
                if len(dif_diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                         ",".join(sorted(dif_diseases)))
                    )
                elif len(dif_diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                         next(iter(dif_diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_disease_by_path1_not_path2"


class DiseaseByMed1NotMed2Template(QATemplate):
    singular_template: str = "What disease is treated by {} but not {}?"
    plural_template: str = "What diseases are treated by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases1 = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med1]
                dif_diseases = diseases1.difference(diseases2)
                if len(dif_diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                         ",".join(sorted(dif_diseases)))
                    )
                elif len(dif_diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                         next(iter(dif_diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_disease_by_med1_not_med2"


class MedicationByDis1NotDis2Template(QATemplate):
    singular_template: str = "What medication treats {} but not {}?"
    plural_template: str = "What medications treat {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            medication1 = graph.disease_medication[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                medication2 = graph.disease_medication[dis2]
                dif_medications = medication1.difference(medication2)
                if len(dif_medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                         ",".join(sorted(dif_medications)))
                    )
                elif len(dif_medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                         next(iter(dif_medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_1_1_medication_by_dis1_not_dis2"


# 2-hop templates
# -------------------------------------


class VectorThruPathByDisTemplate(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes {}?"
    plural_template: str = "What vectors transmit a pathogen which causes {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis in graph.diseases:
            pathogens = graph.disease_pathogen[dis]
            vectors = set()
            for path in pathogens:
                vectors.update(graph.pathogen_vector[path])
            if len(vectors) > 1:
                result['1'].append(
                    (self.plural_template.format(dis),
                        ",".join(sorted(vectors)))
                )
            elif len(vectors) == 1:
                result['0'].append(
                    (self.singular_template.format(dis),
                        next(iter(vectors)))
                )
        return result

    def __str__(self):
        return "t_{}_2_0_vector_thru_path_by_dis"


class DiseaseThruPathByVecTemplate(QATemplate):
    singular_template: str = "What disease is caused by a pathogen which is transmitted by {}?"
    plural_template: str = "What diseases are caused by a pathogen which is transmitted by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec in graph.vectors:
            pathogens = graph.vector_pathogen[vec]
            diseases = set()
            for path in pathogens:
                diseases.update(graph.pathogen_disease[path])
            if len(diseases) > 1:
                result['1'].append(
                    (self.plural_template.format(vec),
                        ",".join(sorted(diseases)))
                )
            elif len(diseases) == 1:
                result['0'].append(
                    (self.singular_template.format(vec),
                        next(iter(diseases)))
                )
        return result

    def __str__(self):
        return "t_{}_2_0_disease_thru_path_by_vec"


class MedicationThruDisByPathTemplate(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by {}?"
    plural_template: str = "What medications treat a disease which is caused by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path in graph.pathogens:
            diseases = graph.pathogen_disease[path]
            medications = set()
            for dis in diseases:
                medications.update(graph.disease_medication[dis])
            if len(medications) > 1:
                result['1'].append(
                    (self.plural_template.format(path),
                        ",".join(sorted(medications)))
                )
            elif len(medications) == 1:
                result['0'].append(
                    (self.singular_template.format(path),
                        next(iter(medications)))
                )
        return result

    def __str__(self):
        return "t_{}_2_0_medication_thru_dis_by_path"


class PathogenThruDisByMedTemplate(QATemplate):
    singular_template: str = "What pathogen causes a disease which is treated by {}?"
    plural_template: str = "What pathogens cause a disease which is treated by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med in graph.medications:
            diseases = graph.medication_disease[med]
            pathogens = set()
            for dis in diseases:
                pathogens.update(graph.disease_pathogen[dis])
            if len(pathogens) > 1:
                result['1'].append(
                    (self.plural_template.format(med),
                        ",".join(sorted(pathogens)))
                )
            elif len(pathogens) == 1:
                result['0'].append(
                    (self.singular_template.format(med),
                        next(iter(pathogens)))
                )
        return result

    def __str__(self):
        return "t_{}_2_0_pathogen_thru_dis_by_med"


class VectorThruPathByDis1AndDis2Template(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes {} and {}?"
    plural_template: str = "What vectors transmit a pathogen which causes {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            pathogens1 = graph.disease_pathogen[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                pathogens2 = graph.disease_pathogen[dis2]
                common_pathogens = pathogens1.intersection(pathogens2)
                vectors = set()
                for path in common_pathogens:
                    vectors.update(graph.pathogen_vector[path])
                if len(vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                            ",".join(sorted(vectors)))
                    )
                elif len(vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                            next(iter(vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_1_vector_thru_path_by_dis1_and_dis2"


class DiseaseThruPathByVec1AndVec2Template(QATemplate):
    singular_template: str = "What disease is caused by a pathogen which is transmitted by {} and {}?"
    plural_template: str = "What diseases are caused by a pathogen which is transmitted by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                common_pathogens = pathogens1.intersection(pathogens2)
                diseases = set()
                for path in common_pathogens:
                    diseases.update(graph.pathogen_disease[path])
                if len(diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                            ",".join(sorted(diseases)))
                    )
                elif len(diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                            next(iter(diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_1_disease_thru_path_by_vec1_and_vec2"


class MedicationThruDisByPath1AndPath2Template(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by {} and {}?"
    plural_template: str = "What medications treat a disease which is caused by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            diseases = graph.pathogen_disease[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                diseases2 = graph.pathogen_disease[path2]
                common_diseases = diseases.intersection(diseases2)
                medications = set()
                for dis in common_diseases:
                    medications.update(graph.disease_medication[dis])
                if len(medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                            ",".join(sorted(medications)))
                    )
                elif len(medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                            next(iter(medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_1_medication_thru_dis_by_path1_and_path2"


class PathogenThruDisByMed1AndMed2Template(QATemplate):
    singular_template: str = "What pathogen causes a disease which is treated by {} and {}?"
    plural_template: str = "What pathogens cause a disease which is treated by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med2]
                common_diseases = diseases.intersection(diseases2)
                pathogens = set()
                for dis in common_diseases:
                    pathogens.update(graph.disease_pathogen[dis])
                if len(pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                            ",".join(sorted(pathogens)))
                    )
                elif len(pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                            next(iter(pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_1_pathogen_thru_dis_by_med1_and_med2"


class VectorThruPathByDis1NotDis2Template(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes {} but not {}?"
    plural_template: str = "What vectors transmit a pathogen which causes {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for dis1 in graph.diseases:
            pathogens1 = graph.disease_pathogen[dis1]
            for dis2 in graph.diseases:
                if dis1 == dis2:
                    continue
                pathogens2 = graph.disease_pathogen[dis2]
                dif_pathogens = pathogens1.difference(pathogens2)
                vectors = set()
                for path in dif_pathogens:
                    vectors.update(graph.pathogen_vector[path])
                if len(vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(dis1, dis2),
                            ",".join(sorted(vectors)))
                    )
                elif len(vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(dis1, dis2),
                            next(iter(vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_2_vector_thru_path_by_dis1_not_dis2"


class DiseaseThruPathByVec1NotVec2Template(QATemplate):
    singular_template: str = "What disease is caused by a pathogen which is transmitted by {} but not {}?"
    plural_template: str = "What diseases are caused by a pathogen which is transmitted by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                dif_pathogens = pathogens1.difference(pathogens2)
                diseases = set()
                for path in dif_pathogens:
                    diseases.update(graph.pathogen_disease[path])
                if len(diseases) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                            ",".join(sorted(diseases)))
                    )
                elif len(diseases) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                            next(iter(diseases)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_2_disease_thru_path_by_vec1_not_vec2"


class MedicationThruDisByPath1NotPath2Template(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by {} but not {}?"
    plural_template: str = "What medications treat a disease which is caused by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for path1 in graph.pathogens:
            diseases = graph.pathogen_disease[path1]
            for path2 in graph.pathogens:
                if path1 == path2:
                    continue
                diseases2 = graph.pathogen_disease[path2]
                dif_diseases = diseases.difference(diseases2)
                medications = set()
                for dis in dif_diseases:
                    medications.update(graph.disease_medication[dis])
                if len(medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(path1, path2),
                            ",".join(sorted(medications)))
                    )
                elif len(medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(path1, path2),
                            next(iter(medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_2_medication_thru_dis_by_path1_not_path2"


class PathogenThruDisByMed1NotMed2Template(QATemplate):
    singular_template: str = "What pathogen causes a disease which is treated by {} but not {}?"
    plural_template: str = "What pathogens cause a disease which is treated by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med2]
                dif_diseases = diseases.difference(diseases2)
                pathogens = set()
                for dis in dif_diseases:
                    pathogens.update(graph.disease_pathogen[dis])
                if len(pathogens) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                            ",".join(sorted(pathogens)))
                    )
                elif len(pathogens) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                            next(iter(pathogens)))
                    )
        return result

    def __str__(self):
        return "t_{}_2_2_pathogen_thru_dis_by_med1_not_med2"


# 3-hop templates
# -------------------------------------


class VectorThruPathThruDisByMedTemplate(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes a disease which is treated by {}?"
    plural_template: str = "What vectors transmit a pathogen which causes a disease which is treated by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med in graph.medications:
            diseases = graph.medication_disease[med]
            pathogens = set()
            for dis in diseases:
                pathogens.update(graph.disease_pathogen[dis])
            vectors = set()
            for path in pathogens:
                vectors.update(graph.pathogen_vector[path])
            if len(vectors) > 1:
                result['1'].append(
                    (self.plural_template.format(med),
                        ",".join(sorted(vectors)))
                )
            elif len(vectors) == 1:
                result['0'].append(
                    (self.singular_template.format(med),
                        next(iter(vectors)))
                )
        return result

    def __str__(self):
        return "t_{}_3_0_vector_thru_path_thru_dis_by_med"


class MedicationThruDisThruPathByVecTemplate(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by a pathogen which is transmitted by {}?"
    plural_template: str = "What medications treat a disease which is caused by a pathogen which is transmitted by {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec in graph.vectors:
            pathogens = graph.vector_pathogen[vec]
            diseases = set()
            for path in pathogens:
                diseases.update(graph.pathogen_disease[path])
            medications = set()
            for dis in diseases:
                medications.update(graph.disease_medication[dis])
            if len(medications) > 1:
                result['1'].append(
                    (self.plural_template.format(vec),
                        ",".join(sorted(medications)))
                )
            elif len(medications) == 1:
                result['0'].append(
                    (self.singular_template.format(vec),
                        next(iter(medications)))
                )
        return result

    def __str__(self):
        return "t_{}_3_0_medication_thru_dis_thru_path_by_vec"


class VectorThruPathThruDisByMed1AndMed2Template(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes a disease which is treated by {} and {}?"
    plural_template: str = "What vectors transmit a pathogen which causes a disease which is treated by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases1 = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med2]
                common_diseases = diseases1.intersection(diseases2)
                pathogens = set()
                for dis in common_diseases:
                    pathogens.update(graph.disease_pathogen[dis])
                vectors = set()
                for path in pathogens:
                    vectors.update(graph.pathogen_vector[path])
                if len(vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                            ",".join(sorted(vectors)))
                    )
                elif len(vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                            next(iter(vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_3_1_vector_thru_path_thru_dis_by_med1_and_med2"


class MedicationThruDisThruPathByVec1AndVec2Template(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by a pathogen which is transmitted by {} and {}?"
    plural_template: str = "What medications treat a disease which is caused by a pathogen which is transmitted by {} and {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                common_pathogens = pathogens1.intersection(pathogens2)
                diseases = set()
                for path in common_pathogens:
                    diseases.update(graph.pathogen_disease[path])
                medications = set()
                for dis in diseases:
                    medications.update(graph.disease_medication[dis])
                if len(medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                            ",".join(sorted(medications)))
                    )
                elif len(medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                            next(iter(medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_3_1_medication_thru_dis_thru_path_by_vec1_and_vec2"


class VectorThruPathThruDisByMed1NotMed2Template(QATemplate):
    singular_template: str = "What vector transmits a pathogen which causes a disease which is treated by {} but not {}?"
    plural_template: str = "What vectors transmit a pathogen which causes a disease which is treated by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for med1 in graph.medications:
            diseases1 = graph.medication_disease[med1]
            for med2 in graph.medications:
                if med1 == med2:
                    continue
                diseases2 = graph.medication_disease[med2]
                dif_diseases = diseases1.difference(diseases2)
                pathogens = set()
                for dis in dif_diseases:
                    pathogens.update(graph.disease_pathogen[dis])
                vectors = set()
                for path in pathogens:
                    vectors.update(graph.pathogen_vector[path])
                if len(vectors) > 1:
                    result['1'].append(
                        (self.plural_template.format(med1, med2),
                            ",".join(sorted(vectors)))
                    )
                elif len(vectors) == 1:
                    result['0'].append(
                        (self.singular_template.format(med1, med2),
                            next(iter(vectors)))
                    )
        return result

    def __str__(self):
        return "t_{}_3_2_vector_thru_path_thru_dis_by_med1_not_med2"


class MedicationThruDisThruPathByVec1NotVec2Template(QATemplate):
    singular_template: str = "What medication treats a disease which is caused by a pathogen which is transmitted by {} but not {}?"
    plural_template: str = "What medications treat a disease which is caused by a pathogen which is transmitted by {} but not {}?"

    def generate(self, graph: MedicalGraph) -> dict[list[(str, str)]]:
        result = {}
        result['0'] = []
        result['1'] = []
        for vec1 in graph.vectors:
            pathogens1 = graph.vector_pathogen[vec1]
            for vec2 in graph.vectors:
                if vec1 == vec2:
                    continue
                pathogens2 = graph.vector_pathogen[vec2]
                dif_pathogens = pathogens1.difference(pathogens2)
                diseases = set()
                for path in dif_pathogens:
                    diseases.update(graph.pathogen_disease[path])
                medications = set()
                for dis in diseases:
                    medications.update(graph.disease_medication[dis])
                if len(medications) > 1:
                    result['1'].append(
                        (self.plural_template.format(vec1, vec2),
                            ",".join(sorted(medications)))
                    )
                elif len(medications) == 1:
                    result['0'].append(
                        (self.singular_template.format(vec1, vec2),
                            next(iter(medications)))
                    )
        return result

    def __str__(self):
        return "t_{}_3_2_medication_thru_dis_thru_path_by_vec1_not_vec2"
