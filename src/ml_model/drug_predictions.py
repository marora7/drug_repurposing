import os
import sys
import argparse
import subprocess
import multiprocessing
import logging
import psutil
from datetime import datetime, timedelta
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from src.utils.db_utils import (
    init_database,
    get_completed_diseases,
    mark_disease_completed,
    record_batch
)
from src.utils.data_utils import load_tsv_mappings
from src.utils.config_utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DrugPredictionController:
    def __init__(self, 
                 entities_file, 
                 model_script='model_test.py', 
                 batch_size=250, 
                 num_workers=14, 
                 config_path=None,
                 memory_limit_gb=None):
        """
        Initialize the controller for drug predictions across multiple disease entities.
        
        Args:
            entities_file (str): Path to the file containing disease entities
            model_script (str): Path to the model testing script to be called
            batch_size (int): Number of diseases to process in each batch
            num_workers (int): Number of parallel processes to run
            config_path (str, optional): Path to the configuration file. If None, will use default location.
            memory_limit_gb (float, optional): Memory limit in GB to prevent OOM errors. If None, will use 90% of available RAM.
        """
        self.entities_file = entities_file

        # Handle model_script path
        if os.path.isabs(model_script):
            # If it's already an absolute path, use it directly
            self.model_script = model_script
        elif model_script == 'src/ml_model/model_test.py':
            # If it's our default path, make it absolute but avoid double-pathing
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_script = os.path.join(base_dir, model_script)
        else:
            # For other relative paths, resolve based on current directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_script = os.path.abspath(os.path.join(base_dir, model_script))
            
        logger.info(f"Using model script at: {self.model_script}")
        
        # Verify the model script exists
        if not os.path.exists(self.model_script):
            logger.warning(f"Model script not found at {self.model_script}. This will cause errors when processing diseases.")

        # Set optimal batch size and workers for 16 CPU system
        self.batch_size = batch_size
        
        # Use 14 workers to leave 2 cores for system processes
        self.num_workers = min(num_workers, multiprocessing.cpu_count() - 2)
        
        # Set memory limit (default to 90% of available memory if not specified)
        if memory_limit_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.memory_limit_gb = total_memory_gb * 0.9
        else:
            self.memory_limit_gb = memory_limit_gb
        
        # Determine config path if not provided
        if config_path is None:
            self.config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        else:
            self.config_path = config_path
        
        # Load configuration using the utility function
        self.config = load_config(self.config_path)
        
        # Get progress tracking database path from config
        self.progress_db_path = self.config.get('ml_model', {}).get('progress_tracking_db', 'data/processed/prediction_progress.db')
        
        # Ensure parent directory for progress DB exists
        os.makedirs(os.path.dirname(self.progress_db_path), exist_ok=True)
        
        # Initialize database for tracking progress
        init_database(self.progress_db_path, self.config)
        
        # Progress tracking attributes
        self.start_time = None
        self.processed_count = 0
        self.avg_time_per_disease = None
        
        logger.info(f"Initialized controller with {self.num_workers} workers and batch size {self.batch_size}")
        logger.info(f"Memory limit set to {self.memory_limit_gb:.2f} GB")
        logger.info(f"Using progress tracking database: {self.progress_db_path}")

    def load_disease_entities(self):
        """
        Load all disease entities from the entities TSV file.
        Returns a list of disease IDs.
        """
        try:
            # Use the existing load_tsv_mappings function to load entities
            entities, _ = load_tsv_mappings(self.entities_file, "entities")
            
            # Filter only disease entities
            disease_entities = [entity for entity in entities if entity.startswith("Disease::")]
            
            logger.info(f"Loaded {len(disease_entities)} unique disease entities")
            return disease_entities
        
        except Exception as e:
            logger.error(f"Error loading disease entities: {str(e)}")
            raise

    def process_disease(self, disease_id):
        """
        Process a single disease by calling the model script.
        
        Args:
            disease_id (str): The ID of the disease to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check memory usage before processing
            current_memory_usage_gb = psutil.virtual_memory().used / (1024 ** 3)
            if current_memory_usage_gb > self.memory_limit_gb:
                logger.warning(f"Memory usage ({current_memory_usage_gb:.2f} GB) exceeds limit ({self.memory_limit_gb:.2f} GB). Postponing disease {disease_id}.")
                time.sleep(10)  # Wait a bit for memory to be freed
                return False
            
            # Command to execute the model script for this disease
            # Based on your model_test.py, we pass the disease name directly as a positional argument
            # No '--output' parameter since your script saves directly to the database
            cmd = [sys.executable, self.model_script, disease_id, '--top_k', str(self.config.get('ml_model', {}).get('top_k', 100))]
            
            logger.info(f"Processing disease {disease_id} with command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            if result.returncode != 0:
                logger.error(f"Error processing disease {disease_id}: {result.stderr}")
                mark_disease_completed(self.progress_db_path, disease_id, 'failed', result.stderr, self.config)
                return False
            
            logger.info(f"Successfully processed disease {disease_id}")
            mark_disease_completed(self.progress_db_path, disease_id, 'completed', None, self.config)
            return True
            
        except Exception as e:
            logger.error(f"Exception processing disease {disease_id}: {str(e)}")
            mark_disease_completed(self.progress_db_path, disease_id, 'failed', str(e), self.config)
            return False

    def process_batch(self, batch_id, disease_batch):
        """
        Process a batch of diseases.
        
        Args:
            batch_id (int): ID of the batch
            disease_batch (list): List of disease IDs to process
            
        Returns:
            tuple: (num_successful, num_failed)
        """
        record_batch(self.progress_db_path, batch_id, len(disease_batch), 'started', None, self.config)
        
        successful = 0
        failed = 0
        
        for disease_id in disease_batch:
            if self.process_disease(disease_id):
                successful += 1
            else:
                failed += 1
                
            # Add a small delay between disease processing to prevent resource contention
            time.sleep(0.5)
        
        status = 'completed' if failed == 0 else 'partial' if successful > 0 else 'failed'
        record_batch(self.progress_db_path, batch_id, len(disease_batch), status, datetime.now().isoformat(), self.config)
        
        return successful, failed

    def worker_function(self, worker_id, job_queue, results_queue):
        """
        Worker function for parallel processing.
        
        Args:
            worker_id (int): ID of the worker
            job_queue (Queue): Queue for jobs to be processed
            results_queue (Queue): Queue for results
        """
        logger.info(f"Worker {worker_id} started")
        while True:
            job = job_queue.get()
            if job is None:  # Poison pill - terminate
                job_queue.task_done()
                break
                
            batch_id, disease_batch = job
            logger.info(f"Worker {worker_id} processing batch {batch_id} with {len(disease_batch)} diseases")
            
            successful, failed = self.process_batch(batch_id, disease_batch)
            results_queue.put((batch_id, successful, failed))
            job_queue.task_done()
            
        logger.info(f"Worker {worker_id} finished")
        results_queue.put(None)  # Signal that this worker is done

    def calculate_eta(self, remaining, processed_so_far):
        """
        Calculate estimated time to completion.
        
        Args:
            remaining (int): Number of remaining items
            processed_so_far (int): Number of items processed
            
        Returns:
            str: Formatted ETA string
        """
        if processed_so_far == 0 or self.start_time is None:
            return "Calculating..."
            
        elapsed = time.time() - self.start_time
        avg_time_per_item = elapsed / processed_so_far
        
        # Store average time for future calculations
        self.avg_time_per_disease = avg_time_per_item
        
        # Estimate remaining time
        remaining_seconds = remaining * avg_time_per_item
        
        # Convert to hours, minutes, seconds
        eta = timedelta(seconds=int(remaining_seconds))
        
        if eta.days > 0:
            return f"{eta.days}d {eta.seconds//3600}h {(eta.seconds//60)%60}m"
        elif eta.seconds // 3600 > 0:
            return f"{eta.seconds//3600}h {(eta.seconds//60)%60}m {eta.seconds%60}s"
        else:
            return f"{(eta.seconds//60)%60}m {eta.seconds%60}s"

    def update_progress_bar(self, pbar, successful, failed, total_remaining):
        """
        Update the progress bar with additional information.
        
        Args:
            pbar (tqdm): Progress bar object
            successful (int): Number of successfully processed items
            failed (int): Number of failed items
            total_remaining (int): Total remaining items
        """
        self.processed_count += (successful + failed)
        
        # Calculate ETA
        eta = self.calculate_eta(total_remaining, self.processed_count)
        
        # Calculate success rate
        success_rate = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 0
        
        # Get current memory usage
        memory_usage = psutil.virtual_memory()
        memory_percent = memory_usage.percent
        memory_used_gb = memory_usage.used / (1024**3)
        
        # Calculate processing rate (items per minute)
        elapsed = time.time() - self.start_time if self.start_time else 0
        items_per_minute = (self.processed_count / elapsed * 60) if elapsed > 0 else 0
        
        # Update progress bar description with all information
        pbar.set_description(
            f"Processing: {self.processed_count}/{self.processed_count + total_remaining} | "
            f"Remaining: {total_remaining} | "
            f"ETA: {eta} | "
            f"Success: {success_rate:.1f}% | "
            f"Rate: {items_per_minute:.1f}/min | "
            f"Mem: {memory_percent:.1f}% ({memory_used_gb:.1f}GB)"
        )
        
        # Update the progress count
        pbar.update(successful + failed)

    def run(self):
        """
        Main method to run the prediction process for all diseases.
        """
        # Log system resource information
        logger.info(f"System resources: {multiprocessing.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.2f} GB RAM")
        
        # Load all disease entities
        all_diseases = self.load_disease_entities()
        
        # Get already completed diseases
        completed_diseases = get_completed_diseases(self.progress_db_path, self.config)
        
        # Filter out already completed diseases
        remaining_diseases = [d for d in all_diseases if d not in completed_diseases]
        logger.info(f"{len(remaining_diseases)} diseases remaining to process out of {len(all_diseases)} total")
        
        if not remaining_diseases:
            logger.info("All diseases have already been processed. Nothing to do.")
            return
        
        # Create batches with optimal size for 16 CPUs and 46GB RAM
        batches = []
        for i in range(0, len(remaining_diseases), self.batch_size):
            batch = remaining_diseases[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch))
        logger.info(f"Created {len(batches)} batches of size up to {self.batch_size}")
        
        # Initialize start time for ETA calculation
        self.start_time = time.time()
        self.processed_count = 0
        
        # Set up multiprocessing with optimized settings
        # Use manager for better memory handling
        with multiprocessing.Manager() as manager:
            job_queue = manager.JoinableQueue()
            results_queue = manager.Queue()
            
            # Start worker processes
            workers = []
            for i in range(self.num_workers):
                p = multiprocessing.Process(
                    target=self.worker_function,
                    args=(i, job_queue, results_queue)
                )
                p.daemon = True  # Enable daemon mode for better cleanup
                p.start()
                workers.append(p)
            
            # Add jobs to the queue
            for batch in batches:
                job_queue.put(batch)
                
            # Add poison pills to stop workers
            for _ in range(self.num_workers):
                job_queue.put(None)
            
            # Process results as they come in
            total_successful = 0
            total_failed = 0
            completed_workers = 0
            total_remaining = len(remaining_diseases)
            
            # Create progress bar with initial status
            with tqdm(total=len(remaining_diseases), desc="Initializing...") as pbar:
                while completed_workers < self.num_workers:
                    try:
                        result = results_queue.get(timeout=300)  # 5-minute timeout
                        if result is None:
                            completed_workers += 1
                            continue
                                
                        batch_id, successful, failed = result
                        total_successful += successful
                        total_failed += failed
                        total_remaining -= (successful + failed)
                        
                        # Update progress bar with additional information
                        self.update_progress_bar(pbar, successful, failed, total_remaining)
                        
                        # Log batch completion
                        current_memory = psutil.virtual_memory()
                        logger.info(f"Batch {batch_id} completed: {successful} successful, {failed} failed. "
                                   f"Memory usage: {current_memory.percent}% ({current_memory.used / (1024**3):.2f} GB)")
                                   
                        # Calculate and log estimated completion time
                        if self.avg_time_per_disease:
                            eta = self.calculate_eta(total_remaining, self.processed_count)
                            logger.info(f"Estimated time to completion: {eta} "
                                       f"(processing rate: {60/self.avg_time_per_disease:.2f} diseases/minute)")
                        
                    except Exception as e:
                        logger.error(f"Error processing results: {str(e)}")
                        
                    # Monitor system resources and adjust if necessary
                    if psutil.virtual_memory().percent > 90:
                        logger.warning("Memory usage is high. Waiting for 30 seconds to allow cleanup.")
                        time.sleep(30)
            
            # Wait for all workers to finish
            for p in workers:
                p.join(timeout=60)  # Wait up to 60 seconds for workers to finish
                if p.is_alive():
                    logger.warning(f"Worker process {p.pid} didn't terminate gracefully, terminating forcefully")
                    p.terminate()
        
        # Calculate total execution time
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Calculate processing rate
        processing_rate = len(remaining_diseases) / elapsed_time * 60 if elapsed_time > 0 else 0
        
        # Final report
        logger.info("=" * 50)
        logger.info("Drug prediction process completed")
        logger.info(f"Total diseases processed: {total_successful + total_failed}")
        logger.info(f"Successfully processed: {total_successful}")
        logger.info(f"Failed to process: {total_failed}")
        logger.info(f"Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
        logger.info(f"Processing rate: {processing_rate:.2f} diseases/minute")
        logger.info("=" * 50)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run drug predictions for multiple disease entities')
    
    parser.add_argument('--entities-file', type=str, required=False,
                        help='Path to the file containing disease entities (if not specified, will be taken from config)')
    
    parser.add_argument('--model-script', type=str, default='model_test.py',
                        help='Path to the model testing script to be called (default: src/ml_model/model_test.py)')
    
    parser.add_argument('--batch-size', type=int, default=250,
                        help='Number of diseases to process in each batch (default: 250)')
    
    parser.add_argument('--num-workers', type=int, default=14,
                        help='Number of parallel processes to run (default: 14)')
    
    parser.add_argument('--config-path', type=str, default='src/config/config.yaml',
                        help='Path to the configuration file (default: src/config/config.yaml)')
    
    parser.add_argument('--memory-limit-gb', type=float, default=40,
                        help='Memory limit in GB to prevent OOM errors (default: 40)')
    
    return parser.parse_args()

def main():
    """
    Main entry point for the drug prediction orchestration script.
    Parses arguments, initializes the controller, and runs the prediction process.
    """
    args = parse_arguments()
    
    # Load config to get entities file path if not provided
    config_path = args.config_path
    config = load_config(config_path)
    
    # If entities file not provided via command line, use the one from config
    entities_file = args.entities_file
    if not entities_file:
        # Construct the path using data_dir and entities_file from config
        data_dir = config.get("data_dir", "data/processed/train")
        entities_filename = config.get("entities_file", "entities.tsv")
        entities_file = os.path.join(data_dir, entities_filename)
        logger.info(f"Using entities file from config: {entities_file}")
    
    start_time = time.time()
    logger.info(f"Starting drug prediction process at {datetime.now().isoformat()}")
    
    controller = DrugPredictionController(
        entities_file=entities_file,
        model_script=args.model_script,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config_path=args.config_path,
        memory_limit_gb=args.memory_limit_gb
    )
    
    controller.run()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    logger.info(f"Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")


if __name__ == "__main__":
    main()