"""
main.py

Changes:
- Correctly unpacks 5 values returned from BenchmarkEvaluator (doc_scores, overall_bleu, overall_cosine, overall_bert, overall_correctness).
- Added comments indicating pipeline steps.
- The 'too many values to unpack' error is fixed by matching the return signature from benchmarking.py.
- Also, you can add or remove calls to toggles if you want to skip certain steps entirely.
"""
# todo: figureing out how to adap this code for different llms/ checking what changes needed for openai
#..... fixing the content that is being sent for each question in inference.py 
# need clean up file whcih cleans up stuff after each iterations
# parallise sending to llms
# get the questions template and code them up for l2,l3,l4,l5
# get api key limit
import os
import logging
import pandas as pd

import config
from graph_builder import main as graph_builder_main
from query_generation import QueryGenerator
from ground_truth import GroundTruthExtractor
from processor import GroundTruthProcessor
from inference import InferencePipeline
from benchmarking import BenchmarkEvaluator

class Pipeline:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self):
        """
        Executes:
          1) Graph Building
          2) Query Generation
          3) Ground Truth Extraction
          4) Ground Truth Processing
          5) Inference
          6) Benchmark
        """
        try:
            # 1. Build Knowledge Graph
            self.logger.info("Building Knowledge Graphs...")
            graph_builder_main()  # uses config inside
            self.logger.info("Knowledge Graph built.")

            # 2. Generate Queries
            self.logger.info("Generating Queries...")
            ontology_file = os.path.join(config.EXTRACTED_CONTENT_DIR, "ontology.ttl")
            qg = QueryGenerator(ontology_file_path=ontology_file, data_directory=config.EXTRACTED_CONTENT_DIR)
            for data_file_path in qg.data_files:
                data_graph = qg.load_graph(data_file_path)
                doc_name   = os.path.basename(data_file_path).replace('.ttl', '')
                qg.L1_person(data_graph, doc_name)
                qg.Level_1_location(data_graph, doc_name)
                qg.Level_1_Roles(data_graph, doc_name)
                qg.L2_person_org(data_graph, doc_name)
            self.logger.info("Queries generated.")

            # 3. Ground Truth
            self.logger.info("Extracting Ground Truth from TTL files...")
            gte = GroundTruthExtractor(ontology_file, config.EXTRACTED_CONTENT_DIR)
            for data_file_path in gte.data_files:
                data_graph = gte.load_graph(data_file_path)
                doc_name   = os.path.basename(data_file_path).replace('.ttl', '')
                gte.Level_1_person_answers(data_graph, doc_name)
                gte.Level_1_location_answers(data_graph, doc_name)
                gte.Level_1_Roles_answers(data_graph, doc_name)
                gte.L2_person_org_answers(data_graph, doc_name)
            self.logger.info("Ground Truth extracted.")

            # 4. Process into a DataFrame
            self.logger.info("Processing Ground Truth into DataFrame...")
            processor = GroundTruthProcessor()
            processor.process_ground_truth_files()
            self.logger.info("Ground Truth DataFrame ready.")

            # 5. Inference
            self.logger.info("Running Inference...")
            inference = InferencePipeline()
            for doc_num in config.DOCUMENT_NUMBERS:
                inference.process_document(doc_num, num_questions=config.QUESTIONS_PER_DOCUMENT)
            self.logger.info("Inference completed.")

            # 6. Benchmark
            self.logger.info("Starting Benchmark Evaluation...")
            # Now we get 5 return values from the updated benchmarking:
            # doc_scores, overall_bleu, overall_cosine, overall_bert, overall_correctness
            doc_scores, overall_bleu, overall_cosine, overall_bert, overall_correctness = \
                BenchmarkEvaluator().evaluate_documents()

            # Build final results DataFrame
            results_df = pd.DataFrame.from_dict(doc_scores, orient='index')
            results_df['document_number']     = results_df.index
            results_df['overall_bleu']        = overall_bleu
            results_df['overall_cosine']      = overall_cosine
            results_df['overall_bert']        = overall_bert
            results_df['overall_correctness'] = overall_correctness

            # Save
            out_csv = os.path.join(config.DATAFRAME_RESULTS_DIR, "benchmark_results.csv")
            results_df.to_csv(out_csv, index=False)
            self.logger.info(f"Benchmark results saved to {out_csv}")

            # Print final
            self.logger.info("Benchmark Results:\n" + str(results_df))
            print("\nPipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_pipeline()
