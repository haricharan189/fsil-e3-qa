import os
from graph_builder import main as graph_builder_main
from query_generation import QueryGenerator
from ground_truth import GroundTruthExtractor
from processor import GroundTruthProcessor
from inference import InferencePipeline
from benchmarking import BenchmarkEvaluator
import logging
import pandas as pd

class Pipeline:
    def __init__(self, api_key):
        """
        Initialize the complete pipeline.
        """
        self.api_key = api_key
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self, document_numbers, questions_per_document=5, final_dir="dataframe_final"):
        """
        Run the complete pipeline for given document numbers.
        
        Args:
            document_numbers (list): List of document numbers to process
            questions_per_document (int): Number of questions to generate per document
            final_dir (str): Directory to store final results
        """
        # Create final directory if it doesn't exist
        os.makedirs(final_dir, exist_ok=True)
        self.final_dir = final_dir
        
        try:
            # 1. Build Knowledge Graphs
            self.logger.info("Starting Knowledge Graph Building...")
            graph_builder_main(json_file_path="48 to 53_hari.json", output_dir="extracted_content")
            self.logger.info("Knowledge Graph Building completed")

            # 2. Generate Queries
            self.logger.info("Starting Query Generation...")
            query_gen = QueryGenerator(
                ontology_file_path="./extracted_content/ontology.ttl",
                data_directory="./extracted_content"
            )
            # Process each document's TTL file
            for data_file_path in query_gen.data_files:
                data_graph = query_gen.load_graph(data_file_path)
                document_name = os.path.basename(data_file_path).replace('.ttl', '')
                
                # Generate Level 1 queries
                query_gen.L1_person(data_graph, document_name)
                query_gen.Level_1_location(data_graph, document_name)
                query_gen.Level_1_Roles(data_graph, document_name)
                
                # Generate Level 2 queries
                query_gen.L2_person_org(data_graph, document_name)
            self.logger.info("Query Generation completed")

            # 3. Generate Ground Truth from TTL files
            self.logger.info("Starting Ground Truth Generation...")
            ground_truth_gen = GroundTruthExtractor(
                ontology_file_path="./extracted_content/ontology.ttl",
                data_directory="./extracted_content"
            )
            self.logger.info("Processing TTL files for ground truth...")
            for data_file_path in ground_truth_gen.data_files:
                data_graph = ground_truth_gen.load_graph(data_file_path)
                document_name = os.path.basename(data_file_path).replace('.ttl', '')
                
                # Generate both Level 1 and Level 2 answers
                ground_truth_gen.Level_1_person_answers(data_graph, document_name)
                ground_truth_gen.Level_1_location_answers(data_graph, document_name)
                ground_truth_gen.Level_1_Roles_answers(data_graph, document_name)
                ground_truth_gen.L2_person_org_answers(data_graph, document_name)
            self.logger.info("Ground Truth Generation completed")

            # 4. Process Ground Truth into DataFrame
            self.logger.info("Starting Ground Truth Processing...")
            processor = GroundTruthProcessor()
            ground_truth_df = processor.process_ground_truth_files()
            self.logger.info("Ground Truth Processing completed")

            # 5. Run Inference
            self.logger.info("Starting Inference...")
            inference = InferencePipeline(self.api_key)
            for doc_num in document_numbers:
                self.logger.info(f"Processing document {doc_num}")
                results = inference.process_document(
                    document_number=doc_num,
                    num_questions=questions_per_document
                )
                if results is not None:
                    self.logger.info(f"Completed inference for document {doc_num}")
            self.logger.info("Inference completed")

            # 6. Run Benchmarking
            self.logger.info("Starting Benchmarking...")
            evaluator = BenchmarkEvaluator()
            doc_scores, overall_f1 = evaluator.evaluate_documents()
            
            # Create and save benchmark results DataFrame
            benchmark_results = pd.DataFrame.from_dict(doc_scores, orient='index')
            benchmark_results['document_number'] = benchmark_results.index
            benchmark_results['overall_f1_score'] = overall_f1
            
            # Save benchmark results
            benchmark_path = os.path.join(self.final_dir, "benchmark_results.csv")
            benchmark_results.to_csv(benchmark_path, index=False)
            self.logger.info(f"Saved benchmark results to {benchmark_path}")
            
            # Print results
            print("\nBenchmarking Results:")
            print("\nDocument-level F1 scores:")
            for doc_num, scores in doc_scores.items():
                print(f"Document {doc_num}:")
                print(f"  Mean F1: {scores['mean_f1']:.4f}")
                print(f"  Questions evaluated: {scores['num_questions']}")
            print(f"\nOverall F1 score: {overall_f1:.4f}")
            
            self.logger.info("Benchmarking completed")
            
            return {
                'doc_scores': doc_scores,
                'overall_f1': overall_f1
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # TogetherAI API key
    API_KEY = "api_key"
    
    # List of document numbers to process
    DOCUMENT_NUMBERS = [215, 216, 217, 218, 219]
    
    # Questions per document
    QUESTIONS_PER_DOC = 4
    
    # Run pipeline
    pipeline = Pipeline(API_KEY)
    try:
        results = pipeline.run_pipeline(
            document_numbers=DOCUMENT_NUMBERS,
            questions_per_document=QUESTIONS_PER_DOC,
            final_dir="dataframe_results"
        )
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
