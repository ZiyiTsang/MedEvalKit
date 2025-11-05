
"""
Filter Datasets: Universal Medical Data Filtering Framework

This script provides a comprehensive filtering system for medical benchmark datasets.
It uses the new Filter class from filter_source package to filter samples based on
specific medical categories using either keyword matching or GPT judgment.

Key Features:
- Universal filtering framework for any medical category
- Two-stage filtering: keyword matching + GPT judgment  
- Support for multiple datasets with different structures
- Image path to PIL object conversion
- Checkpointing and batch processing for large files
- Standardized output format
- Jinja2 template-based prompt system
- Customizable filtering criteria
- Extensible keyword system with predefined medical categories

Usage:
    python filter_datasets.py --input mm_benchmarks.json --output filtered_samples.json --category dental --keyword_only
    python filter_datasets.py --input mm_benchmarks.json --output filtered_samples.json --category cardiac --gpt_judgment --template image_text

Available Categories:
- dental: Dental and oral health related keywords
- cardiac: Cardiovascular related keywords  
- cancer: Cancer and oncology related keywords
- neurological: Neurological related keywords
- respiratory: Respiratory related keywords
- general: General medical terms
"""

import os
import json
import argparse
from typing import List, Dict, Any

# Import the new Filter framework
from filter_source import Filter, get_keywords_for_category, GPT5V


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Filter medical samples from multimodal benchmarks")
    
    # Input/output paths
    parser.add_argument('--input', type=str, default='mm_benchmarks.json',
                       help="Input multimodal benchmarks json file path")
    parser.add_argument('--output', type=str, default='filtered_samples.json',
                       help="Output filtered samples json file path")
    
    # Medical category selection
    parser.add_argument('--category', type=str, default='general',
                       choices=['dental', 'cardiac', 'cancer', 'neurological', 'respiratory', 'general'],
                       help="Medical category for filtering")
    
    # Filtering mode selection
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--keyword_only', action='store_true', default=False,
                            help="Use only keyword filtering (fast but less accurate)")
    filter_group.add_argument('--gpt_judgment', action='store_true', default=False,
                            help="Use GPT judgment after keyword filtering (slower but more accurate)")
    
    # GPT configuration
    parser.add_argument('--model_name', type=str, default='gpt-4',
                       help="GPT model name for judgment")
    parser.add_argument('--api_key', type=str, required=False,
                       help="API key for GPT service (required for GPT judgment)")
    parser.add_argument('--api_url', type=str, required=False,
                       help="API URL for GPT service (required for GPT judgment)")
    
    # Template selection
    parser.add_argument('--template', type=str, default='image_text',
                       choices=['text', 'image', 'combined', 'image_text', 'debug'],
                       help="GPT template type: 'text', 'image', 'combined', 'image_text', or 'debug'")
    
    # Processing configuration
    parser.add_argument('--batch_size', type=int, default=100,
                       help="Batch size for processing samples")
    parser.add_argument('--max_workers', type=int, default=10,
                       help="Maximum number of worker threads")
    
    # Image compression configuration
    parser.add_argument('--max_image_dimension', type=int, default=1024,
                       help="Maximum image dimension for compression (width or height)")
    parser.add_argument('--compress_on_413_only', action='store_true', default=True,
                       help="Only compress images when encountering 413 errors (default: True)")
    parser.add_argument('--always_compress', action='store_true', default=False,
                       help="Always compress images before upload (overrides --compress_on_413_only)")
    
    # Dataset filtering
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help="Specific datasets to process (space-separated list). If not specified, process all datasets.")
    parser.add_argument('--skip_datasets', type=str, nargs='+', default=['SLAKE'],
                       help="Datasets to skip (space-separated list). Default: SLAKE")
    
    # Output options
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help="Dry run mode - show statistics without saving output")
    
    # Test mode (only available in dry run)
    parser.add_argument('--test_mode', action='store_true', default=False,
                       help="Test mode - process only one batch per dataset (requires --dry_run)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        print("Please provide a valid input file using --input parameter.")
        return 1
    
    # Determine filtering mode
    if args.gpt_judgment:
        print(f"Using GPT judgment mode for {args.category} filtering")
        use_gpt = True
    elif args.keyword_only:
        print(f"Using keyword-only filtering mode for {args.category} filtering")
        use_gpt = False
    else:
        print("Error: No filtering mode specified.")
        print("Please specify either --keyword_only or --gpt_judgment.")
        print("Use --help for more information.")
        return 1
    
    # Initialize GPT client if needed
    gpt_client = None
    if use_gpt:
        # Validate API credentials
        if not args.api_key:
            print("Error: --api_key is required for GPT judgment mode")
            print("Please provide an API key using --api_key parameter")
            return 1
        if not args.api_url:
            print("Error: --api_url is required for GPT judgment mode")
            print("Please provide an API URL using --api_url parameter")
            return 1
            
        print(f"Initializing GPT client with model: {args.model_name}")
        print(f"  API URL: {args.api_url}")
        # Determine compression strategy
        compress_on_413_only = args.compress_on_413_only and not args.always_compress
        if args.always_compress:
            print(f"  Image compression: ALWAYS compress (max dimension: {args.max_image_dimension})")
        else:
            print(f"  Image compression: Only on 413 errors (max dimension: {args.max_image_dimension})")
            
        gpt_client = GPT5V(
            api_key=args.api_key,
            api_url=args.api_url,
            model_name=args.model_name,
            max_image_dimension=args.max_image_dimension,
            compress_on_413_only=compress_on_413_only
        )
    
    # Get keywords for the specified category
    keywords = get_keywords_for_category(args.category)
    print(f"Using {args.category} keywords: {len(keywords)} keywords loaded")
    
    # Initialize filter
    filter_instance = Filter(
        input_json=args.input,
        output_json=args.output,
        gpt_client=gpt_client,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        template_type=args.template,
        datasets=args.datasets,
        skip_datasets=args.skip_datasets,
        max_image_dimension=args.max_image_dimension,
        compress_on_413_only=args.compress_on_413_only and not args.always_compress,
        keywords=keywords,
        category_name=args.category
    )
    
    # Validate test mode usage
    if args.test_mode and not args.dry_run:
        print("Error: --test_mode can only be used with --dry_run")
        print("Please use --test_mode together with --dry_run")
        return 1
    
    try:
        # Filter samples
        filtered_samples, all_data = filter_instance.filter_samples(dry_run=args.dry_run, test_mode=args.test_mode)
        
        # Show statistics
        print(f"\n=== FILTERING RESULTS ===")
        print(f"Total {args.category}-related samples found: {len(filtered_samples)}")
        
        # Calculate total samples based on data format
        if isinstance(all_data, dict):
            # Dictionary format: {"dataset_name": [samples...]}
            total_samples = sum(len(samples) for samples in all_data.values())
        elif isinstance(all_data, (list, tuple)):
            # List format: [samples...]
            total_samples = len(all_data)
        else:
            total_samples = 0
            
        percentage = (len(filtered_samples) / total_samples * 100) if total_samples > 0 else 0
        print(f"Total samples processed: {total_samples}")
        print(f"Percentage of {args.category}-related samples: {percentage:.2f}%")
        
        # Show breakdown by dataset
        dataset_counts = {}
        for sample in filtered_samples:
            dataset = sample.get("dataset", "unknown")
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print(f"\nBreakdown by dataset:")
        for dataset, count in dataset_counts.items():
            # Calculate dataset total based on data format
            if isinstance(all_data, dict):
                dataset_total = len(all_data.get(dataset, []))
            elif isinstance(all_data, (list, tuple)):
                # For list format, count samples with matching dataset field
                dataset_total = sum(1 for sample in all_data if isinstance(sample, dict) and sample.get('dataset') == dataset)
            else:
                dataset_total = 0
                
            dataset_percentage = (count / dataset_total * 100) if dataset_total > 0 else 0
            print(f"  {dataset}: {count}/{dataset_total} samples ({dataset_percentage:.2f}%)")
        
        filter_instance.save_results()
        if args.dry_run:
            print(f"\nDry run completed - results saved to: {filter_instance.json_output}")
        else:
            print(f"\nFinal results saved to: {filter_instance.json_output}")
        
        # Show sample preview
        if filtered_samples:
            print(f"\nSample preview (first 3 samples):")
            for i, sample in enumerate(filtered_samples[:3]):
                print(f"\nSample {i+1}:")
                print(f"  Dataset: {sample.get('dataset', 'unknown')}")
                print(f"  Prompt: {sample.get('prompt', '')[:100]}...")
                print(f"  Answer: {sample.get('answer', '')}")
                if 'img_path' in sample and sample['img_path']:
                    print(f"  Images: {len(sample['img_path'])} image(s)")
                if args.gpt_judgment:
                    print(f"  GPT Judgment: {sample.get('GPT', 'None')}")
                else:
                    print(f"  Keyword Match: {sample.get('match', 'None')}")
                    print(f"  Matched Keywords: {sample.get('matched_keywords', [])}")

        # Clean up state file
        if os.path.exists(filter_instance.state_file):
            os.remove(filter_instance.state_file)
        
    except Exception as e:
        print(f"Error during filtering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
   