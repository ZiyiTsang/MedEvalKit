"""
Filter: Universal Medical Data Filtering Framework

This module provides a general-purpose filtering framework for medical benchmark datasets.
It supports both keyword-based and GPT-based filtering with customizable templates.

Key Features:
- Two-stage filtering: keyword matching + GPT judgment
- Support for multiple datasets with different structures
- Image path to PIL object conversion
- Checkpointing and batch processing for large files
- Standardized output format
- Jinja2 template-based prompt system
- Customizable filtering criteria
- Extensible keyword system with predefined medical categories

Usage:
    from filter_source.basic import Filter
    from filter_source.keywords import get_keywords_for_category
    
    # Use predefined dental keywords
    keywords = get_keywords_for_category('dental')
    
    filter = Filter(
        input_json="mm_benchmarks.json",
        output_json="filtered_samples.json",
        gpt_client=gpt_client,
        keywords=keywords,
        template_type="text"  # text, image, combined, image_text, debug
    )
    results = filter.filter_samples()
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
import random
import jinja2
import tempfile
import shutil

# Import keywords module
from .keywords import get_keywords_for_category, get_all_keywords


class Filter:
    """
    Universal filter class for processing multimodal benchmark data
    """
    
    def __init__(self, input_json: str, output_json: str, gpt_client=None,
                 batch_size: int = 1000, max_workers: int = 1,
                 template_type: str = "text", datasets: List[str] = None,
                 skip_datasets: List[str] = None,
                 max_image_dimension: int = 1024, compress_on_413_only: bool = True,
                 keywords: set = None, category_name: str = "general"):
        """
        Initialize the filter
        
        Args:
            input_json: Path to input mm_benchmarks.json file
            output_json: Path to output filtered samples.json file (can be full path or relative path)
            gpt_client: GPT client for judgment
            batch_size: Number of samples to process in each batch
            max_workers: Maximum number of worker threads
            template_type: Type of template to use ("text", "image", "combined", "image_text", "debug")
            datasets: List of datasets to process, None means all datasets
            skip_datasets: List of datasets to skip
            max_image_dimension: Maximum dimension for image compression
            compress_on_413_only: Whether to compress images only on 413 errors
            keywords: Set of keywords for filtering, default uses dental keywords
            category_name: Name of the filtering category for logging
        """
        self.input_json = input_json
        self.output_json = output_json
        self.gpt_client = gpt_client
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.template_type = template_type
        self.datasets = datasets  # List of datasets to process, None means all datasets
        self.skip_datasets = skip_datasets or []  # List of datasets to skip
        self.max_image_dimension = max_image_dimension
        self.compress_on_413_only = compress_on_413_only
        self.category_name = category_name
        
        # Initialize keywords based on category if not provided
        if keywords is None:
            keywords = get_keywords_for_category(category_name)
        self.keywords = keywords
        
        # Initialize Jinja2 template loader
        self.template_loader = jinja2.DictLoader({})
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self._load_templates()
        
        # Initialize dataset processors
        self.processors = {
            'MMMU-Medical-test': MMMUProcessor(input_json),
            'MMMU-Medical-val': MMMUProcessor(input_json),
            'PMC_VQA': PMCVQAProcessor(input_json),
            'OmniMedVQA': OmniMedVQAProcessor(input_json),
            'MedXpertQA-MM': MedXpertQAProcessor(input_json),
            'VQA_RAD': VQARADProcessor(input_json),
            'SLAKE': SLAKEProcessor(input_json),
            'PATH_VQA': PATHVQAProcessor(input_json),
            'MedFrameQA': MedFrameQAProcessor(input_json)
        }
        
        # Parse output path to determine output directory and filename
        output_dir = os.path.dirname(output_json) if os.path.dirname(output_json) else '.'
        output_filename = os.path.basename(output_json)
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        # State tracking for checkpointing
        self.state_file = os.path.join(output_dir, output_filename.replace('.json', '_state.json'))
        
        # JSON output file (must be defined before _load_state)
        self.json_output = os.path.join(output_dir, output_filename)
        
        # Results storage
        self.filtered_samples = []
        
        # Image storage configuration - images will be saved in 'images' subdirectory of output directory
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Now load state (after json_output is defined)
        self.state = self._load_state()
    
    def _load_templates(self):
        """Load Jinja2 templates from the prompt directory"""
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompt')
        
        if os.path.exists(prompt_dir):
            # Load templates from files
            template_files = {
                'image': 'image_judgment.j2',
                'text': 'text_judgment.j2',
                'combined': 'image_judgment.j2',  # Uses image template
                'image_text': 'image_text_judgment.j2',
                'debug': 'debug_description.j2'
            }
            
            for template_type, filename in template_files.items():
                template_path = os.path.join(prompt_dir, filename)
                if os.path.exists(template_path):
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            template_content = f.read()
                        self.template_loader.mapping[template_type] = template_content
                    except Exception as e:
                        print(f"Warning: Failed to load template {template_type}: {e}")
        
        # Fallback templates if files don't exist
        if not self.template_loader.mapping:
            self._create_fallback_templates()
    
    def _create_fallback_templates(self):
        """Create fallback templates if file loading fails"""
        self.template_loader.mapping = {
            'image': """Please analyze the provided medical image and determine if it shows {{ category_name }}-related content.
Answer only "yes" or "no" or "none".

Criteria for judgment:
- If you can see the image clearly and it shows {{ category_name }}-related structures, answer "yes"
- If you can see the image clearly but it does NOT show {{ category_name }}-related structures, answer "no"
- If the you cannot see the image at all or the image is not visible, answer "none".

IMPORTANT:
- Base your judgment ONLY on the image content, not on any accompanying text.""",
            
            'text': """Please determine if the following medical question is related to {{ category_name }}. Answer only "yes" or "no".

Question: {{ prompt }}

Criteria for judgment:
- Does it involve {{ category_name }}-related topics?
- Does it involve medical content related to {{ category_name }}?

If related, answer "yes"; if not related, answer "no".""",
            
            'image_text': """Please analyze both the provided medical image and the question to determine if they are related to {{ category_name }}. Answer only "yes", "no", or "none".
----------------------------------------------------------------
Question:
{{ prompt }}
----------------------------------------------------------------

Criteria for judgment:
- If you can see the image clearly and both the image AND the question are clearly related to {{ category_name }}, answer "yes"
- If you can see the image clearly but either the image OR the question is not clearly related to {{ category_name }}, answer "no"
- If you cannot see the image at all or the image is not visible, answer "none"

IMPORTANT:
- Base your judgment on both the image content and the question text.""",
            
            'debug': """Please describe what you see in the provided medical image(s).

If there is only one image, describe it in one sentence focusing on the main content and notable features.

If there are multiple images, please describe each image separately in the following format:
Image 1: [description of first image]
Image 2: [description of second image]
Image 3: [description of third image]
etc.

Focus on the medical content and any notable features in each image."""
        }
    
    def _load_state(self) -> Dict[str, Any]:
        """Load processing state from state file and existing results"""
        # Default state
        state = {
            "processed_datasets": [],
            "current_dataset": None,
            "current_sample_index": 0,
            "processed_samples": 0,
            "failed_samples": [],
            "filtered_samples_count": 0,
            "start_time": None,
            "last_update": None
        }
        
        # Load state file if exists
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                    state.update(loaded_state)
                print(f"Loaded state from {self.state_file}")
            except Exception as e:
                print(f"Warning: Failed to load state file: {e}")
        
        # Load existing results if JSON file exists
        if os.path.exists(self.json_output):
            try:
                with open(self.json_output, 'r', encoding='utf-8') as f:
                    existing_samples = json.load(f)
                    self.filtered_samples = existing_samples
                    state["filtered_samples_count"] = len(existing_samples)
                    print(f"Loaded {len(existing_samples)} existing samples from {self.json_output}")
            except Exception as e:
                print(f"Warning: Failed to load existing results: {e}")
        
        return state
    
    def _save_state(self):
        """Save current processing state to file"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save state file: {e}")
    
    def _is_related(self, prompt: str, images: List[Image.Image] = None, dry_run=False, verbose=False) -> tuple[bool, list[str], Optional[bool]]:
        """
        Filter related samples using either keyword matching OR GPT judgment
        
        Args:
            prompt: Text prompt from the sample
            images: List of PIL Image objects from the sample
            dry_run: Whether in dry run mode (print GPT output)
            
        Returns:
            Tuple of (is_related, matched_keywords, gpt_result)
        """
        # Use either keyword filtering OR GPT judgment, not both
        if self.gpt_client:
            # GPT judgment mode - ignore keyword matching
            gpt_result = self._gpt_judgment(prompt, images, dry_run=dry_run, verbose=verbose)
            return gpt_result, [], gpt_result
        else:
            # Keyword-only mode
            is_keyword_match, matched_keywords = self._keyword_filter(prompt)
            return is_keyword_match, matched_keywords, None
    
    def _keyword_filter(self, text: str) -> tuple[bool, list[str]]:
        """
        Fast keyword-based filtering with whole word matching
        
        Args:
            text: Text to check for related keywords
            
        Returns:
            Tuple of (is_related, matched_keywords)
        """
        text_lower = text.lower()
        matched_keywords = []
        for keyword in self.keywords:
            keyword_lower = keyword.lower()
            # Use regex for whole word matching
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, text_lower):
                matched_keywords.append(keyword)
        return len(matched_keywords) > 0, matched_keywords
    
    def _get_judgment_prompt(self, prompt: str, images: List[Image.Image] = None) -> str:
        """
        Get the appropriate judgment prompt based on template type
        
        Args:
            prompt: Text prompt from the sample
            images: List of PIL Image objects from the sample
            
        Returns:
            Formatted judgment prompt string
        """
        try:
            template = self.template_env.get_template(self.template_type)
            return template.render(prompt=prompt, category_name=self.category_name)
        except Exception as e:
            print(f"Warning: Failed to load template '{self.template_type}', using fallback: {e}")
            # Fallback to text template if specific template fails
            try:
                template = self.template_env.get_template('text')
                return template.render(prompt=prompt, category_name=self.category_name)
            except:
                # Ultimate fallback - simple prompt
                return f"Please determine if the following question is related to {self.category_name}. Answer only 'yes' or 'no'.\n\nQuestion: {prompt}\n\nIf related, answer 'yes'; if not related, answer 'no'."
    
    def _gpt_judgment(self, prompt: str, images: List[Image.Image] = None, dry_run=False, verbose=False) -> bool:
        """
        GPT-based judgment for relevance
        
        Args:
            prompt: Text prompt from the sample
            images: List of PIL Image objects from the sample
            dry_run: Whether in dry run mode (print GPT output)
            
        Returns:
            True if GPT judges as related, False otherwise (including API failures)
        """
        try:
            # Get judgment prompt based on template type
            judgment_prompt = self._get_judgment_prompt(prompt, images)
            
            # Determine judgment type for logging
            template_type_names = {
                "text": "TEXT-BASED",
                "image": "IMAGE-BASED",
                "combined": "COMBINED (auto-select)",
                "image_text": "IMAGE-TEXT COMBINED",
                "debug": "DEBUG (description)"
            }
            judgment_type = template_type_names.get(self.template_type, f"CUSTOM ({self.template_type})")
            
            # Print image path details in verbose mode
            if verbose and images:
                print(f"\n=== IMAGE PATH DETAILS ===")
                for i, img in enumerate(images):
                    if hasattr(img, 'filename'):
                        print(f"Image {i+1}: {img.filename}")
                    else:
                        print(f"Image {i+1}: [PIL Image object, no filename]")
                print(f"=== END IMAGE PATH DETAILS ===\n")
            
            # Call GPT with appropriate parameters
            if images and self.template_type in ["image", "combined", "image_text", "debug"]:
                response = self.gpt_client.call(judgment_prompt, images=images, verbose=verbose, dry_run=dry_run)
            else:
                response = self.gpt_client.call(judgment_prompt, verbose=verbose, dry_run=dry_run)
            
            # Parse response - handle both string and dict responses
            if isinstance(response, dict):
                # Handle error responses from GPT API
                if 'error' in response:
                    error_msg = response['error'].get('message', 'Unknown GPT error')
                    error_code = response['error'].get('code', 'unknown')
                    print(f"❌ GPT API error: {error_msg} (code: {error_code})")
                    
                    # Handle content filtering errors specifically
                    if 'content_filter' in error_code or 'content management policy' in error_msg:
                        print(f"⚠️  Content filtered by GPT API, skipping sample")
                        return False
                    
                    # For other API errors, skip the sample
                    print(f"⚠️  GPT API error, skipping sample")
                    return False
                else:
                    # Unexpected dict response, treat as error
                    print(f"Warning: Unexpected GPT response format: {response}")
                    return False
            
            # Normal string response
            response_lower = response.strip().lower()
            
            # For debug template, print the description but don't filter based on it
            if self.template_type == "debug":
                print(f"🎯 DEBUG IMAGE DESCRIPTION: {response}")
                # In debug mode, we want to see all images but not filter them as related
                # Return False to avoid filtering non-related images
                return False
            
            # Handle "none" response - treat same as "no" but print info in dry_run/verbose mode
            if response_lower == 'none':
                if dry_run or verbose:
                    print(f"⚠️  GPT returned 'none' - image not visible or unclear")
                    print(f"   Prompt: {prompt[:100]}...")
                    if images:
                        for i, img in enumerate(images):
                            if hasattr(img, 'filename'):
                                print(f"   Image {i+1} Path: {img.filename}")
                    print(f"   GPT Response: {response}")
                # Treat "none" same as "no" - return False
                is_related = False
            else:
                # Only return True if GPT explicitly says yes/true/是
                # All other responses (including descriptions, errors, or unclear answers) should return False
                is_related = response_lower in ['yes', 'true', '是']
            
            # Print verbose details in dry run mode
            if dry_run and verbose:
                print(f"\n=== GPT JUDGMENT DETAILS ===")
                print(f"Judgment Type: {judgment_type}")
                print(f"Sample Prompt: {prompt[:200]}...")
                print(f"Images Available: {len(images) if images else 0}")
                if images:
                    for i, img in enumerate(images):
                        if hasattr(img, 'filename'):
                            print(f"Image {i+1} Path: {img.filename}")
                        else:
                            print(f"Image {i+1}: [PIL Image object]")
                print(f"GPT Prompt Used: {judgment_prompt[:200]}...")
                print(f"GPT Response: {response}")
                print(f"=== END GPT JUDGMENT ===\n")
            
            # Only print in dry_run mode when GPT says YES
            if dry_run and is_related:
                print(f"🎯 GPT JUDGMENT: {self.category_name.upper()}-RELATED SAMPLE FOUND!")
                print(f"   Prompt: {prompt[:100]}...")
                if images:
                    for i, img in enumerate(images):
                        if hasattr(img, 'filename'):
                            print(f"   Image {i+1} Path: {img.filename}")
                print(f"   Response: {response}")
            
            return is_related
            
        except Exception as e:
            print(f"❌ GPT judgment failed: {e}")
            # Skip sample on any exception
            return False
        finally:
            # Ensure images are cleaned up after GPT judgment
            if images:
                self._cleanup_images(images)
    
    def _cleanup_images(self, images: List[Image.Image]):
        """Clean up PIL Image objects to free file handles"""
        for img in images:
            try:
                if hasattr(img, 'close'):
                    img.close()
            except Exception as e:
                # Silently ignore cleanup errors
                pass
    
    def filter_samples(self, dry_run=False, test_mode=False) -> List[Dict[str, Any]]:
        """
        Main filtering function
        
        Args:
            dry_run: Whether in dry run mode
            test_mode: Whether in test mode (process only one batch per dataset)
            
        Returns:
            List of related samples in standardized format
        """
        print(f"Loading {self.category_name} benchmark data from JSON...")
        
        # Force single worker in dry run mode
        if dry_run:
            print("DRY RUN MODE: Forcing single worker for debugging")
            self.max_workers = 1
        
        # Adjust worker count for non-dry run mode
        if not dry_run and self.max_workers > self.batch_size:
            print(f"WARNING: max_workers ({self.max_workers}) > batch_size ({self.batch_size})")
            print(f"Reducing max_workers to batch_size ({self.batch_size})")
            self.max_workers = self.batch_size
        
        # Limit max_workers to prevent too many file handles being opened simultaneously
        # Each worker can open multiple images, so we need to be conservative
        max_recommended_workers = min(10, self.max_workers)  # Limit to 10 workers max
        if self.max_workers > max_recommended_workers:
            print(f"WARNING: Reducing max_workers from {self.max_workers} to {max_recommended_workers} to prevent file handle exhaustion")
            self.max_workers = max_recommended_workers
        
        # Print current filtering mode
        if self.gpt_client:
            print(f"FILTERING MODE: GPT judgment with '{self.template_type}' template")
        else:
            print("FILTERING MODE: Keyword matching only (GPT judgment disabled)")
            
        # Print test mode status
        if test_mode:
            print("TEST MODE: Processing only one batch per dataset")
        
        try:
            # Load JSON file
            with open(self.input_json, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # Debug: print the type and structure of loaded data
            print(f"Loaded data type: {type(all_data)}")
            
            # Detect and handle different JSON formats
            if isinstance(all_data, dict):
                print(f"Data format: DICTIONARY (dataset_name -> samples)")
                print(f"Data keys: {list(all_data.keys())}")
                for key, value in list(all_data.items())[:3]:  # Show first 3 datasets
                    print(f"Dataset '{key}': {len(value)} samples")
                data_format = "dict"
                
            elif isinstance(all_data, (list, tuple)):
                print(f"Data format: LIST (samples with dataset field)")
                print(f"Data length: {len(all_data)} samples")
                if len(all_data) > 0:
                    # Check if samples have dataset field
                    first_sample = all_data[0]
                    if isinstance(first_sample, dict) and 'dataset' in first_sample:
                        print(f"First sample dataset: {first_sample.get('dataset', 'unknown')}")
                    else:
                        print(f"Warning: List format but samples don't have 'dataset' field")
                data_format = "list"
                
            else:
                print(f"Error: Unknown data format: {type(all_data)}")
                return []
                
        except Exception as e:
            print(f"Error loading input file: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Process data based on format
        if data_format == "dict":
            # Original dictionary format: {"dataset_name": [samples...]}
            total_samples = sum(len(samples) for samples in all_data.values())
            print(f"Found {len(all_data)} datasets with {total_samples} total samples")
            print(f"Using {self.max_workers} workers for parallel processing")
            print(f"Batch size: {self.batch_size}")
            
            # Process datasets
            for dataset_name, samples in all_data.items():
                # Skip datasets not in the specified list
                if self.datasets and dataset_name not in self.datasets:
                    print(f"Skipping dataset not in filter list: {dataset_name}")
                    continue
                
                # Skip datasets in the skip list
                if self.skip_datasets and dataset_name in self.skip_datasets:
                    print(f"Skipping dataset in skip list: {dataset_name}")
                    continue

                # Check if dataset is already processed
                if dataset_name in self.state["processed_datasets"]:
                    print(f"Skipping already processed dataset: {dataset_name}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing dataset: {dataset_name} ({len(samples)} samples)")
                print(f"{'='*60}")
                self.state["current_dataset"] = dataset_name
                self._save_state()
                
                # Get appropriate processor
                processor = self.processors.get(dataset_name, DefaultProcessor(self.input_json))
                
                # Calculate number of batches
                num_batches = (len(samples) + self.batch_size - 1) // self.batch_size
                
                # In test mode, only process one batch
                if test_mode:
                    num_batches = 1
                    print(f"TEST MODE: Processing only 1 batch of {self.batch_size} samples")
                else:
                    print(f"Processing {len(samples)} samples in {num_batches} batches")
                
                # Determine starting point for this dataset
                start_index = 0
                if self.state["current_dataset"] == dataset_name:
                    start_index = self.state["current_sample_index"]
                    print(f"Resuming from batch starting at sample {start_index}")
                
                # Process samples in batches
                batch_count = 0
                for batch_idx, batch_start in enumerate(range(start_index, len(samples), self.batch_size)):
                    batch_end = min(batch_start + self.batch_size, len(samples))
                    batch_samples = samples[batch_start:batch_end]
                    
                    # Calculate actual batch number (starting from 1) based on start_index
                    actual_batch_num = (batch_start // self.batch_size) + 1
                    print(f"  Batch {actual_batch_num}/{num_batches}: samples {batch_start}-{batch_end}")
                    self._process_batch(batch_samples, processor, dataset_name, dry_run=dry_run)
                    
                    # Save results and state after each batch (always save, regardless of dry_run)
                    self.save_results(incremental=True)
                    
                    # Update state
                    self.state["current_sample_index"] = batch_end
                    self.state["processed_samples"] += len(batch_samples)
                    self._save_state()
                    
                    # In test mode, break after first batch
                    batch_count += 1
                    if test_mode and batch_count >= 1:
                        print(f"  TEST MODE: Completed first batch, skipping remaining batches")
                        break
                
                # Mark dataset as processed
                self.state["processed_datasets"].append(dataset_name)
                self.state["current_dataset"] = None
                self.state["current_sample_index"] = 0
                self._save_state()
                
                # Show examples from this dataset
                self._show_dataset_examples(dataset_name)
                
                print(f"✓ Completed dataset: {dataset_name}")
                
        elif data_format == "list":
            # New list format: [samples...] with each sample containing dataset field
            total_samples = len(all_data)
            print(f"Found {total_samples} samples in list format")
            print(f"Using {self.max_workers} workers for parallel processing")
            print(f"Batch size: {self.batch_size}")
            
            # Group samples by dataset
            dataset_samples = {}
            for sample in all_data:
                if isinstance(sample, dict) and 'dataset' in sample:
                    dataset_name = sample['dataset']
                    if dataset_name not in dataset_samples:
                        dataset_samples[dataset_name] = []
                    dataset_samples[dataset_name].append(sample)
                else:
                    print(f"Warning: Sample missing 'dataset' field: {sample}")
            
            print(f"Grouped into {len(dataset_samples)} datasets: {list(dataset_samples.keys())}")
            
            # Process datasets
            for dataset_name, samples in dataset_samples.items():
                # Skip datasets not in the specified list
                if self.datasets and dataset_name not in self.datasets:
                    print(f"Skipping dataset not in filter list: {dataset_name}")
                    continue
                
                # Skip datasets in the skip list
                if self.skip_datasets and dataset_name in self.skip_datasets:
                    print(f"Skipping dataset in skip list: {dataset_name}")
                    continue

                # Check if dataset is already processed
                if dataset_name in self.state["processed_datasets"]:
                    print(f"Skipping already processed dataset: {dataset_name}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing dataset: {dataset_name} ({len(samples)} samples)")
                print(f"{'='*60}")
                self.state["current_dataset"] = dataset_name
                self._save_state()
                
                # Get appropriate processor
                processor = self.processors.get(dataset_name, DefaultProcessor(self.input_json))
                
                # Calculate number of batches
                num_batches = (len(samples) + self.batch_size - 1) // self.batch_size
                
                # In test mode, only process one batch
                if test_mode:
                    num_batches = 1
                    print(f"TEST MODE: Processing only 1 batch of {self.batch_size} samples")
                else:
                    print(f"Processing {len(samples)} samples in {num_batches} batches")
                
                # Determine starting point for this dataset
                start_index = 0
                if self.state["current_dataset"] == dataset_name:
                    start_index = self.state["current_sample_index"]
                    print(f"Resuming from batch starting at sample {start_index}")
                
                # Process samples in batches
                batch_count = 0
                for batch_idx, batch_start in enumerate(range(start_index, len(samples), self.batch_size)):
                    batch_end = min(batch_start + self.batch_size, len(samples))
                    batch_samples = samples[batch_start:batch_end]
                    
                    # Calculate actual batch number (starting from 1) based on start_index
                    actual_batch_num = (batch_start // self.batch_size) + 1
                    print(f"  Batch {actual_batch_num}/{num_batches}: samples {batch_start}-{batch_end}")
                    self._process_batch(batch_samples, processor, dataset_name, dry_run=dry_run)
                    
                    # Save results and state after each batch (always save, regardless of dry_run)
                    self.save_results(incremental=True)
                    
                    # Update state
                    self.state["current_sample_index"] = batch_end
                    self.state["processed_samples"] += len(batch_samples)
                    self._save_state()
                    
                    # In test mode, break after first batch
                    batch_count += 1
                    if test_mode and batch_count >= 1:
                        print(f"  TEST MODE: Completed first batch, skipping remaining batches")
                        break
                
                # Mark dataset as processed
                self.state["processed_datasets"].append(dataset_name)
                self.state["current_dataset"] = None
                self.state["current_sample_index"] = 0
                self._save_state()
                
                # Show examples from this dataset
                self._show_dataset_examples(dataset_name)
                
                print(f"✓ Completed dataset: {dataset_name}")
        
        print(f"\nFiltering completed. Found {len(self.filtered_samples)} {self.category_name}-related samples")
        return self.filtered_samples, all_data
    
    def _process_batch(self, batch_samples: List[Dict], processor, dataset_name: str, dry_run=False):
        """Process a batch of samples"""
        if dry_run:
            # Single-threaded processing for debugging with strict error handling
            print(f"    Dry run: processing {len(batch_samples)} samples sequentially")
            print(f"    Forcing single worker in dry run mode")
            for i, sample in enumerate(batch_samples):
                try:
                    result = self._process_single_sample(sample, processor, dataset_name, dry_run=dry_run)
                    if result:
                        self.filtered_samples.append(result)
                        self.state["filtered_samples_count"] += 1
                except Exception as e:
                    # In dry_run mode, re-raise the exception to terminate immediately
                    raise Exception(f"Error processing sample {i} in dry_run mode: {e}")
        else:
            # Multi-threaded processing for production
            print(f"    Parallel processing {len(batch_samples)} samples with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for sample in batch_samples:
                    future = executor.submit(self._process_single_sample_wrapper, sample, processor, dataset_name, dry_run=False)
                    futures.append(future)
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures),
                                 desc=f"  {dataset_name}", leave=False):
                    try:
                        result = future.result()
                        if result:
                            self.filtered_samples.append(result)
                            self.state["filtered_samples_count"] += 1
                    except Exception as e:
                        print(f"Error processing sample: {e}")
    
    def _process_single_sample_wrapper(self, sample: Dict, processor, dataset_name: str, dry_run=False) -> Optional[Dict]:
        """Wrapper method for parallel processing that calls the instance method"""
        return self._process_single_sample(sample, processor, dataset_name, dry_run=dry_run)
    
    def _process_single_sample(self, sample: Dict, processor, dataset_name: str, dry_run=False) -> Optional[Dict]:
        """Process a single sample"""
        images = []
        try:
            # Extract prompt
            prompt = processor.extract_prompt(sample)
            if not prompt:
                return None
            
            # Extract images for GPT judgment
            images = processor.process_images(sample, dry_run=dry_run)
            
            # Check if related and get matched keywords and GPT result
            is_related, matched_keywords, gpt_result = self._is_related(prompt, images, dry_run=dry_run, verbose=dry_run)
            if not is_related:
                # Clean up images if sample is not related
                self._cleanup_images(images)
                return None
            
            # Process the sample
            standardized_sample = processor.process_sample(sample)
            if not standardized_sample:
                # Clean up images if sample processing fails
                self._cleanup_images(images)
                return None
            
            # Add dataset information and filtering results
            standardized_sample["dataset"] = dataset_name
            standardized_sample["matched_keywords"] = matched_keywords
            
            # Add GPT field: True/False/None
            standardized_sample["GPT"] = gpt_result
            
            # Add match field: True/False/None according to requirements
            if self.gpt_client:
                # GPT judgment mode: match should be None (not enabled)
                standardized_sample["match"] = None
            else:
                # Keyword-only mode: match should be True/False
                standardized_sample["match"] = bool(matched_keywords)
                # Clear matched_keywords if match is False
                if not standardized_sample["match"]:
                    standardized_sample["matched_keywords"] = []
            
            if 'images' not in standardized_sample and 'img_path' in standardized_sample:
                standardized_sample['images'] = standardized_sample['img_path']
                del standardized_sample['img_path']
            
            # Save images and replace image objects with relative paths
            if 'images' in standardized_sample and standardized_sample['images']:
                image_paths = []
                for i, img in enumerate(standardized_sample['images']):
                    if hasattr(img, 'size'):  # PIL Image object
                        # Generate unique filename
                        timestamp = int(time.time() * 1000)
                        random_suffix = random.randint(1000, 9999)
                        content_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
                        filename = f"{dataset_name}_{content_hash}_{timestamp}_{random_suffix}_{i}.jpg"
                        image_path = os.path.join(self.images_dir, filename)
                        
                        # Convert image to RGB mode if necessary before saving as JPEG
                        # Handle different image modes: P, RGBA, etc.
                        if img.mode in ['P', 'RGBA', 'LA', 'PA']:
                            # Convert palette or alpha channel images to RGB
                            if 'A' in img.mode:
                                # For images with alpha channel, create white background
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'RGBA':
                                    background.paste(img, mask=img.split()[-1])
                                else:
                                    background.paste(img, mask=img)
                                img = background
                            else:
                                # For palette images, convert directly
                                img = img.convert('RGB')
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save image
                        img.save(image_path, format='JPEG')
                        
                        # Store relative path
                        relative_path = os.path.join('images', filename)
                        image_paths.append(relative_path)
                    else:
                        # Already a path string
                        image_paths.append(img)
                
                # Rename images field to img_path
                standardized_sample["img_path"] = image_paths
                del standardized_sample["images"]
            
            # Clean up temporary images after processing
            self._cleanup_images(images)
            return standardized_sample
            
        except Exception as e:
            # Ensure images are cleaned up even if an error occurs
            self._cleanup_images(images)
            if dry_run:
                raise Exception(f"Error processing sample in dry_run mode: {e}")
            else:
                print(f"Error processing sample: {e}")
                return None
    
    def _show_dataset_examples(self, dataset_name: str):
        """Show examples from the current dataset"""
        # Get samples from this dataset
        dataset_samples = [sample for sample in self.filtered_samples if sample.get("dataset") == dataset_name]
        
        if not dataset_samples:
            print(f"  No {self.category_name}-related samples found in {dataset_name}")
            return
        
        print(f"\n  Found {len(dataset_samples)} {self.category_name}-related samples in {dataset_name}")
        print(f"  Showing {min(3, len(dataset_samples))} examples:")
        
        # Show up to 3 examples
        for i, sample in enumerate(dataset_samples[:3]):
            print(f"\n  Example {i+1}:")
            print(f"    Dataset: {sample.get('dataset', 'unknown')}")
            print(f"    Prompt: {sample.get('prompt', '')}")
            print(f"    Answer: {sample.get('answer', '')}")
            print(f"    GT Content: {sample.get('gt_content', '')}")
            if 'img_path' in sample and sample['img_path']:
                print(f"    Images: {len(sample['img_path'])} image(s)")
            print(f"    GPT: {sample.get('GPT', 'None')}")
            print(f"    match: {sample.get('match', 'None')}")
            if self.gpt_client:
                print(f"    Filtering Mode: GPT judgment")
            else:
                print(f"    Filtering Mode: Keyword matching")
                print(f"    Matched Keywords: {sample.get('matched_keywords', [])}")
            print(f"    All fields: {list(sample.keys())}")
    
    def save_results(self, incremental=False):
        """Save filtered samples to JSON file"""
        try:
            # Always save JSON file, even if no samples (creates empty array)
            with open(self.json_output, 'w', encoding='utf-8') as f:
                json.dump(self.filtered_samples, f, indent=2, ensure_ascii=False)
            
            if incremental:
                print(f"Incrementally saved {len(self.filtered_samples)} {self.category_name} samples to {self.json_output}")
            else:
                print(f"Saved {len(self.filtered_samples)} {self.category_name} samples to {self.json_output}")
                
        except Exception as e:
            print(f"Error saving results: {e}")


# Import the time module for timestamp generation
import time


class DatasetProcessor:
    """Base class for dataset-specific processing"""
    
    def __init__(self, input_json: str = None):
        self.input_json = input_json
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single sample and return standardized format
        
        Returns:
            Standardized sample with keys: prompt, answer, dataset, gt_content
        """
        try:
            # Extract required fields
            prompt = self.extract_prompt(sample)
            answer = self.extract_answer(sample)
            gt_content = self.extract_gt_content(sample)
            
            if not prompt:
                return None
            
            # Process images
            images = self.process_images(sample)
            
            # Build standardized sample
            standardized = {
                "prompt": prompt,
                "answer": answer,
                "dataset": sample.get("dataset", ""),
                "gt_content": gt_content
            }
            
            # Add images if available
            if images:
                standardized["img_path"] = images
            
            return standardized
            
        except Exception as e:
            print(f"Error in process_sample: {e}")
            return None
    
    def extract_prompt(self, sample: Dict[str, Any]) -> str:
        """Extract prompt text from sample"""
        # JSON format uses "prompt" field directly
        if "prompt" in sample:
            return sample["prompt"]
        else:
            return ""
    
    def extract_answer(self, sample: Dict[str, Any]) -> str:
        """Extract answer from sample"""
        # Try different field names
        if "answer" in sample:
            return str(sample["answer"])
        elif "gt_answer" in sample:
            return str(sample["gt_answer"])
        elif "label" in sample:
            return str(sample["label"])
        elif "correct_answer" in sample:
            return str(sample["correct_answer"])
        elif "gt_content" in sample:
            return str(sample["gt_content"])
        else:
            return ""
    
    def extract_gt_content(self, sample: Dict[str, Any]) -> str:
        """Extract ground truth content from sample"""
        # Try different field names
        if "gt_content" in sample:
            return str(sample["gt_content"])
        elif "correct_choice" in sample and "index2ans" in sample:
            # For multiple choice questions, get the full text
            correct_choice = sample["correct_choice"]
            index2ans = sample["index2ans"]
            if correct_choice in index2ans:
                return str(index2ans[correct_choice])
        elif "answer" in sample:
            return str(sample["answer"])
        else:
            return ""
    
    def process_images(self, sample: Dict[str, Any], dry_run=False) -> List[Image.Image]:
        """Convert image paths to PIL Image objects from img_path field"""
        images = []
        
        # JSON format uses "img_path" field
        if "img_path" in sample and isinstance(sample["img_path"], list):
            for img_path in sample["img_path"]:
                if isinstance(img_path, str):
                    # String path - load the image
                    # Get the directory of the input JSON file to resolve relative image paths
                    if self.input_json:
                        input_dir = os.path.dirname(self.input_json)
                        # Build the full image path relative to the input JSON file location
                        full_img_path = os.path.join(input_dir, img_path)
                    else:
                        # Fallback: use relative path from current directory
                        full_img_path = img_path
                    
                    if os.path.exists(full_img_path) and os.path.isfile(full_img_path):
                        try:
                            # Use context manager to ensure file is properly closed
                            with Image.open(full_img_path) as img:
                                # Load the image data and close the file handle immediately
                                img.load()
                                # Store the original filename in the image object for debugging
                                img.filename = full_img_path
                                images.append(img)
                            if dry_run:
                                print(f"    Loaded image: {full_img_path}")
                        except Exception as e:
                            error_msg = f"ERROR: Failed to load image {full_img_path}: {e}"
                            # If any image fails to load, immediately abort the program
                            print(f"CRITICAL ERROR: {error_msg}")
                            raise Exception(f"Image loading failed: {error_msg}")
                    else:
                        error_msg = f"ERROR: Image file not found: {full_img_path}"
                        print(f"CRITICAL ERROR: {error_msg}")
                        raise Exception(f"Image file not found: {error_msg}")
                elif hasattr(img_path, 'size'):
                    # Already a PIL Image object
                    images.append(img_path)
        
        if dry_run and images:
            print(f"    Total images loaded: {len(images)}")
        
        return images


class MMMUProcessor(DatasetProcessor):
    """Processor for MMMU datasets"""
    
    def process_images(self, sample: Dict[str, Any], dry_run=False) -> List[Image.Image]:
        """Convert image paths to PIL Image objects from img_path field"""
        # MMMU datasets in JSON format use the same img_path field as others
        return super().process_images(sample, dry_run=dry_run)


class PMCVQAProcessor(DatasetProcessor):
    def process_images(self, sample: Dict[str, Any], dry_run=False) -> List[Image.Image]:
        return super().process_images(sample, dry_run=dry_run)


class OmniMedVQAProcessor(DatasetProcessor):
    """Processor for OmniMedVQA dataset"""
    pass  # Uses base implementation


class MedXpertQAProcessor(DatasetProcessor):
    """Processor for MedXpertQA-MM dataset"""
    pass  # Uses base implementation


class VQARADProcessor(DatasetProcessor):
    """Processor for VQA_RAD dataset"""
    pass  # Uses base implementation


class SLAKEProcessor(DatasetProcessor):
    """Processor for SLAKE dataset"""
    pass  # Uses base implementation


class PATHVQAProcessor(DatasetProcessor):
    """Processor for PATH_VQA dataset"""
    pass  # Uses base implementation


class MedFrameQAProcessor(DatasetProcessor):
    """Processor for MedFrameQA dataset"""
    pass  # Uses base implementation


class DefaultProcessor(DatasetProcessor):
    """Default processor for unknown datasets"""
    pass  # Uses base implementation