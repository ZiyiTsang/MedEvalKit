#!/usr/bin/env python3
"""
Simplified Multimodal Benchmark Data Collection Script

Usage:
    python collect_multimodal_data_simple.py --output multimodal_benchmarks.json
    
    # Test mode: Only process 5 samples per dataset
    python collect_multimodal_data_simple.py --test_mode --output test_multimodal_benchmarks.json
    
    # Process specific datasets only
    python collect_multimodal_data_simple.py --datasets PMC_VQA VQA_RAD --output specific_datasets.json
    
Available datasets: MMMU-Medical-val, PMC_VQA, OmniMedVQA, MedXpertQA-MM, VQA_RAD, SLAKE, PATH_VQA, MedFrameQA-MM
"""

import os
import sys
import json
import argparse
import shutil
import hashlib
from typing import List, Dict, Any, Union
from tqdm import tqdm
from PIL import Image








class MultimodalDataCollector:
    """Collector for multimodal benchmark data"""
    
    def __init__(self, datasets_path: str = "hf", images_dir: str = "images"):
        """
        Initialize the data collector
        
        Args:
            datasets_path: Path to datasets directory or "hf" for HuggingFace datasets
            images_dir: Directory to save processed images (relative to project root)
        """
        self.datasets_path = datasets_path
        self.images_dir = images_dir
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Only include multimodal datasets that contain images
        self.multimodal_datasets = [
            "MMMU-Medical-val", "PMC_VQA", "OmniMedVQA", "MedXpertQA-MM", "VQA_RAD", "SLAKE","PATH_VQA", "MedFrameQA-MM"
        ]
        
    def collect_data(self, test_mode: bool = False, target_datasets: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect data from multimodal benchmarks
        
        Args:
            test_mode: If True, only process 5 samples per dataset for testing
            target_datasets: List of specific datasets to process, None for all datasets
            
        Returns:
            Dictionary with dataset names as keys and lists of standardized samples as values
        """
        all_data = {}
        
        # Determine which datasets to process
        if target_datasets is None:
            datasets_to_process = self.multimodal_datasets
        else:
            datasets_to_process = [ds for ds in target_datasets if ds in self.multimodal_datasets]
            # Check for invalid dataset names
            invalid_datasets = [ds for ds in target_datasets if ds not in self.multimodal_datasets]
            if invalid_datasets:
                print(f"Warning: Invalid dataset names: {', '.join(invalid_datasets)}")
                print(f"Available datasets: {', '.join(self.multimodal_datasets)}")
        
        if not datasets_to_process:
            print("No valid datasets to process")
            return all_data
        
        print("Starting multimodal data collection...")
        if test_mode:
            print("🚀 TEST MODE: Only processing 5 samples per dataset")
        print(f"Target datasets: {', '.join(datasets_to_process)}")
        
        for dataset_name in tqdm(datasets_to_process, desc="Processing datasets"):
            try:
                dataset_samples = self._process_dataset(dataset_name, test_mode)
                all_data[dataset_name] = dataset_samples
                print(f"✓ {dataset_name}: collected {len(dataset_samples)} samples")
            except Exception as e:
                print(f"✗ {dataset_name}: failed to process - {str(e)}")
                continue
                
        total_samples = sum(len(samples) for samples in all_data.values())
        print(f"\nTotal samples collected: {total_samples}")
        return all_data
    
    def _process_dataset(self, dataset_name: str, test_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Process a single dataset and extract samples
        
        Args:
            dataset_name: Name of the dataset to process
            test_mode: If True, only process 5 samples for testing
            
        Returns:
            List of processed samples from the dataset
        """
        # Handle MMMU datasets specially
        if dataset_name in ["MMMU-Medical-test", "MMMU-Medical-val"]:
            return self._process_mmmu_dataset(dataset_name, test_mode)
        
        # Import dataset class dynamically to avoid circular imports
        dataset_class = self._import_dataset_class(dataset_name)
        if dataset_class is None:
            raise ValueError(f"Could not import dataset class for {dataset_name}")
        
        # Initialize dataset with None model (we don't need model for data collection)
        model = None
        dataset_path = self.datasets_path if self.datasets_path != "hf" else None
        output_path = "./temp_output"  # Temery output path
        
        # Create dataset instance
        if dataset_name in ["MedXpertQA-Text", "MedXpertQA-MM"]:
            _, split = dataset_name.split("-")
            dataset = dataset_class(model, dataset_path, output_path, split)
        else:
            dataset = dataset_class(model, dataset_path, output_path)
        
        # Load data
        samples = dataset.load_data()
        
        # Limit to 5 samples in test mode
        if test_mode:
            samples = samples[:5]
            print(f"  Test mode: Processing only {len(samples)} samples from {dataset_name}")
        
        # Process each sample with index to ensure unique hashes
        processed_samples = []
        for idx, sample in enumerate(samples):
            processed_sample = self._standardize_sample(sample, dataset_name, dataset.dataset_path, idx)
            processed_samples.append(processed_sample)
            
        return processed_samples
    
    def _process_mmmu_dataset(self, dataset_name: str, test_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Process MMMU dataset using the data collector
        
        Args:
            dataset_name: Name of the MMMU dataset
            test_mode: If True, only process 5 samples for testing
            
        Returns:
            List of processed samples from MMMU dataset
        """
        try:
            from utils.MMMU.data_collector import load_mmmu_data
            
            # Parse dataset name
            _, subset, split = dataset_name.split("-")
            
            # Map split names to MMMU supported splits
            split_mapping = {
                "test": "test",
                "val": "validation",  # MMMU uses "validation" not "val"
                "validation": "validation"
            }
            
            if split not in split_mapping:
                print(f"Warning: Unsupported split '{split}' for MMMU dataset {dataset_name}")
                return []
            
            mapped_split = split_mapping[split]
            
            # Set dataset path
            dataset_path = self.datasets_path if self.datasets_path != "hf" else "MMMU/MMMU"
            
            # Load data
            samples = load_mmmu_data(dataset_path, mapped_split, subset)
            
            # Limit to 5 samples in test mode
            if test_mode:
                samples = samples[:5]
                print(f"  Test mode: Processing only {len(samples)} samples from {dataset_name}")
            
            # Process each sample with index to ensure unique hashes
            processed_samples = []
            for idx, sample in enumerate(samples):
                processed_sample = self._standardize_sample(sample, dataset_name, dataset_path, idx)
                processed_samples.append(processed_sample)
                
            return processed_samples
            
        except Exception as e:
            print(f"Warning: Failed to process MMMU dataset {dataset_name}: {e}")
            return []
    
    def _import_dataset_class(self, dataset_name: str):
        """
        Dynamically import the dataset class to avoid circular imports
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset class or None if import fails
        """
        try:
            if dataset_name == "PATH_VQA":
                from utils.PATH_VQA.PATH_VQA import PATH_VQA
                return PATH_VQA
            elif dataset_name == "PMC_VQA":
                from utils.PMC_VQA.PMC_VQA import PMC_VQA
                return PMC_VQA
            elif dataset_name == "VQA_RAD":
                from utils.VQA_RAD.VQA_RAD import VQA_RAD
                return VQA_RAD
            elif dataset_name == "SLAKE":
                from utils.SLAKE.SLAKE import SLAKE
                return SLAKE
            elif dataset_name == "OmniMedVQA":
                from utils.OmniMedVQA.OmniMedVQA import OmniMedVQA
                return OmniMedVQA
            elif dataset_name in ["MedXpertQA-Text", "MedXpertQA-MM"]:
                from utils.MedXpertQA.MedXpertQA import MedXpertQA
                return MedXpertQA
            elif dataset_name == "IU_XRAY":
                from utils.IU_XRAY.IU_XRAY import IU_XRAY
                return IU_XRAY
            elif dataset_name == "MIMIC_CXR":
                from utils.MIMIC_CXR.MIMIC_CXR import MIMIC_CXR
                return MIMIC_CXR
            elif dataset_name == "MedFrameQA-MM":
                from utils.MedFrameQA.MedFrameQA import MedFrameQA
                return MedFrameQA
            else:
                return None
        except ImportError as e:
            print(f"Warning: Could not import dataset class for {dataset_name}: {e}")
            return None
    
    def _standardize_sample(self, sample: Dict[str, Any], dataset_name: str, dataset_path: str, sample_index: int = None) -> Dict[str, Any]:
        """
        标准化样本，只保留关键字段
        
        Args:
            sample: Original sample from dataset
            dataset_name: Name of the source dataset
            dataset_path: Path to the dataset directory
            sample_index: Index of the sample in the dataset (for unique hashing)
            
        Returns:
            Standardized sample dictionary with required fields
        """

        img_paths = self._extract_image_paths(sample, dataset_name, dataset_path, sample_index)
        
        # 创建只包含关键字段的新样本
        standardized_sample = {
            "answer": "",
            "gt_answer": "",
            "choice": "",
            "prompt": "",
            "img_path": img_paths
        }
        
        # 填充 answer 字段
        if "answer" in sample:
            standardized_sample["answer"] = sample["answer"]
        elif "gt_answer" in sample:
            standardized_sample["answer"] = sample["gt_answer"]
        elif "label" in sample:
            standardized_sample["answer"] = sample["label"]
        elif "correct_answer" in sample:
            standardized_sample["answer"] = sample["correct_answer"]
        elif "gt_content" in sample:
            standardized_sample["answer"] = sample["gt_content"]
        
        # 填充 gt_answer 字段
        if "gt_answer" in sample:
            standardized_sample["gt_answer"] = sample["gt_answer"]
        elif "gt_content" in sample:
            standardized_sample["gt_answer"] = sample["gt_content"]
        else:
            standardized_sample["gt_answer"] = standardized_sample["answer"]
        
        # 填充 choice 字段
        if "choice" in sample:
            standardized_sample["choice"] = sample["choice"]
        elif "choices" in sample:
            standardized_sample["choice"] = sample["choices"]
        
        # 填充 prompt 字段
        if "prompt" in sample:
            standardized_sample["prompt"] = sample["prompt"]
        elif "messages" in sample and isinstance(sample["messages"], dict):
            standardized_sample["prompt"] = sample["messages"].get("prompt", "")
        elif "final_input_prompt" in sample:
            standardized_sample["prompt"] = sample["final_input_prompt"]
        elif "question" in sample:
            standardized_sample["prompt"] = sample["question"]
        standardized_sample["prompt"].replace("<image 1>", " ").strip()

        
        return standardized_sample
    
    def _process_image(self, image_source, dataset_name: str, sample_hash: str) -> str:
        """
        Process an image source (path string or image object) and save to images directory
        
        Args:
            image_source: Image source (path string or PIL Image object)
            dataset_name: Name of the dataset for naming
            sample_hash: Hash value of the sample for unique naming
            
        Returns:
            Relative path to the saved image
        """
        try:
            # Determine file extension
            if isinstance(image_source, str):
                _, ext = os.path.splitext(image_source)
                if not ext:
                    ext = '.jpg'
            else:
                ext = '.jpg'
            
            # Generate unique filename using hash
            filename = f"{dataset_name}_{sample_hash}{ext}"
            relative_path = os.path.join(self.images_dir, filename)
            absolute_path = os.path.abspath(relative_path)
            
            # For datasets where multiple samples may share the same image object,
            # we need to always reprocess the image to ensure each sample gets its own copy
            # Remove the cache check to force reprocessing
            # if os.path.exists(absolute_path):
            #     return relative_path
            
            # Save the image
            if isinstance(image_source, str):
                # Copy the file
                shutil.copy2(image_source, absolute_path)
            else:
                # Save the image object with proper format handling
                if hasattr(image_source, 'save'):
                    try:
                        # Try to save with original format
                        image_source.save(absolute_path)
                    except Exception as save_error:
                        # If saving fails, convert to RGB and save as JPEG
                        try:
                            if image_source.mode in ('RGBA', 'LA', 'P'):
                                # Convert RGBA to RGB for JPEG compatibility
                                rgb_image = image_source.convert('RGB')
                                rgb_image.save(absolute_path, 'JPEG')
                            else:
                                # For other modes, try saving as PNG
                                image_source.save(absolute_path, 'PNG')
                        except Exception as convert_error:
                            print(f"Warning: Failed to save image after conversion: {convert_error}")
                            return ""
                else:
                    # Reopen and save if it's a file-like object
                    try:
                        with Image.open(image_source.filename) as img:
                            if img.mode in ('RGBA', 'LA', 'P'):
                                rgb_img = img.convert('RGB')
                                rgb_img.save(absolute_path, 'JPEG')
                            else:
                                img.save(absolute_path)
                    except Exception as file_error:
                        print(f"Warning: Failed to save image from file: {file_error}")
                        return ""
            
            return relative_path
            
        except Exception as e:
            print(f"Warning: Failed to process image for {dataset_name}: {str(e)}")
            return ""
    
    def _generate_sample_hash(self, sample: Dict[str, Any], dataset_name: str) -> str:
        """
        Generate a unique hash for the sample based on its content
        
        Args:
            sample: Sample dictionary
            dataset_name: Name of the dataset
            
        Returns:
            Hash string (first 8 characters of MD5 hash)
        """
        # Create a string representation of the sample for hashing
        sample_str = f"{dataset_name}_{sample.get('prompt', '')}_{sample.get('answer', '')}_{sample.get('gt_answer', '')}"
        
        # Generate MD5 hash and take first 8 characters for shorter filename
        hash_obj = hashlib.md5(sample_str.encode('utf-8'))
        return hash_obj.hexdigest()[:10]
    
    def _generate_sample_hash_with_index(self, sample: Dict[str, Any], dataset_name: str, sample_index: int) -> str:
        """
        Generate a unique hash for the sample based on its content and index
        This ensures uniqueness even for samples with identical content
        
        Args:
            sample: Sample dictionary
            dataset_name: Name of the dataset
            sample_index: Index of the sample in the dataset
            
        Returns:
            Hash string (first 8 characters of MD5 hash)
        """
        # Create a string representation including the sample index
        # This guarantees uniqueness even for identical samples
        sample_str = f"{dataset_name}_{sample_index}_{sample.get('prompt', '')}_{sample.get('answer', '')}_{sample.get('gt_answer', '')}"
        
        # Generate MD5 hash and take first 8 characters for shorter filename
        hash_obj = hashlib.md5(sample_str.encode('utf-8'))
        return hash_obj.hexdigest()[:10]
    
    def _extract_image_paths(self, sample: Dict[str, Any], dataset_name: str, dataset_path: str, sample_index: int = None) -> List[str]:
        """
        Extract image paths from sample, process images, and return relative paths
        
        Args:
            sample: Sample dictionary
            dataset_name: Name of the dataset
            dataset_path: Path to dataset directory
            sample_index: Index of the sample in the dataset (for unique hashing)
            
        Returns:
            List of relative image paths (empty list if no images)
        """
        img_relative_paths = []
        
        try:
            # Generate unique hash for this sample using sample index to ensure uniqueness
            # This prevents duplicate image paths even for samples with identical content
            sample_hash = self._generate_sample_hash_with_index(sample, dataset_name, sample_index)
            
            # Handle different dataset image field conventions
            if dataset_name in ["OmniMedVQA"]:
                # OmniMedVQA uses image_path field
                if "image_path" in sample and sample["image_path"]:
                    relative_path = self._process_image(sample["image_path"], dataset_name, sample_hash)
                    if relative_path:
                        img_relative_paths.append(relative_path)
                            

                # Also check for img_name field for backward compatibility
                elif "img_name" in sample and sample["img_name"]:
                    img_path = os.path.join(dataset_path, "imgs", sample["img_name"])
                    if os.path.exists(img_path):
                        relative_path = self._process_image(img_path, dataset_name, sample_hash)
                        if relative_path:
                            img_relative_paths.append(relative_path)
                    else:
                        print(f"Warning: SLAKE image path does not exist: {img_path}")
                        
            elif dataset_name in ["IU_XRAY", "MIMIC_CXR"]:
                # X-ray datasets use image list field
                if "image" in sample and sample["image"]:
                    for i, img_rel_path in enumerate(sample["image"]):
                        img_abs_path = os.path.join(dataset_path, "images", img_rel_path)
                        if os.path.exists(img_abs_path):
                            # For multiple images in one sample, append index to hash
                            multi_image_hash = f"{sample_hash}_{i}"
                            relative_path = self._process_image(img_abs_path, dataset_name, multi_image_hash)
                            if relative_path:
                                img_relative_paths.append(relative_path)
                            
                        
            elif "MMMU" in dataset_name:
                # MMMU datasets use image field
                if "image" in sample and sample["image"]:
                    relative_path = self._process_image(sample["image"], dataset_name, sample_hash)
                    if relative_path:
                        img_relative_paths.append(relative_path)

            else:
                if "messages" in sample and isinstance(sample["messages"], dict):
                    image_source = sample["messages"].get("image")
                    if not image_source:
                        image_source=sample["messages"].get("images")
                    if type(image_source) != list:
                        image_source = [image_source]
                    for i, img in enumerate(image_source):
                            # For datasets like VQA_RAD where multiple samples may share the same image,
                            # we need to ensure each sample gets a unique image path by including the sample hash
                            multi_image_hash = f"{sample_hash}_{i}"
                            relative_path = self._process_image(img, dataset_name, multi_image_hash)
                            if relative_path:
                                img_relative_paths.append(relative_path)
                        

        
        except Exception as e:
            print(f"Warning: Failed to extract image paths for {dataset_name}: {str(e)}")

        if len(img_relative_paths) == 0:
            print(f"Warning: No valid images found in sample from {dataset_name}")
        
        return img_relative_paths
    
    def save_data(self, data: Dict[str, List[Dict[str, Any]]], output_file: str):
        """
        保存收集的数据到JSON文件，只保留关键字段
        
        Args:
            data: Dictionary with dataset names as keys and lists of samples as values
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if len(data) == 0:
            print("Warning: No data to save")
            return

        # 确保所有样本只包含关键字段
        filtered_data = {}
        for dataset_name, samples in data.items():
            filtered_samples = []
            for sample in samples:
                # 只保留指定字段
                filtered_sample = {
                    "answer": sample.get("answer", ""),
                    "gt_answer": sample.get("gt_answer", ""),
                    "choice": sample.get("choice", ""),
                    "prompt": sample.get("prompt", ""),
                    "img_path": sample.get("img_path", [])
                }
                filtered_samples.append(filtered_sample)
            filtered_data[dataset_name] = filtered_samples
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {output_file} (JSON format)")
        print(f"Images saved to: {self.images_dir}")




def main_with_validation():
    """Main function with automatic validation after collection"""
    parser = argparse.ArgumentParser(description="Collect multimodal benchmark data")
    parser.add_argument("--output", "-o", default="./total2/mm_benchmarks.json",
                       help="Output file path (default: ./total2/mm_benchmarks.json)")
    parser.add_argument("--datasets_path", "-d", default="hf",
                       help="Path to datasets directory or 'hf' for HuggingFace (default: hf)")
    parser.add_argument("--images_dir", "-i", default="./total2/images",
                       help="Directory to save processed images (default: ./total2/images)")
    parser.add_argument("--test_mode", "-t", action="store_true",
                       help="Test mode: Only process 5 samples per dataset")
    parser.add_argument("--datasets", nargs="+",
                       help="Specific datasets to process (space-separated). If not provided, process all datasets.")
    
    args = parser.parse_args()
    
    # Check if output file already exists
    if not os.path.exists(args.output):
        # Set environment variables to avoid API key errors. No use of real API keys in this script.
        os.environ["api_key"] = "dummy_key" 
        os.environ["use_llm_judge"] = "False"
        
        # Initialize collector
        collector = MultimodalDataCollector(datasets_path=args.datasets_path, images_dir=args.images_dir)
        
        try:
            # Collect data
            all_data = collector.collect_data(test_mode=args.test_mode, target_datasets=args.datasets)
            
            # Save data
            collector.save_data(all_data, args.output)
            
            print(f"\nData collection completed successfully!")
            total_samples = sum(len(samples) for samples in all_data.values())
            print(f"Total samples: {total_samples}")
            print(f"Output file: {args.output}")
            
            
        except Exception as e:
            print(f"Error during data collection: {str(e)}")
            sys.exit(1)

    print(f"Output file {args.output} already exists. Skipping data collection.")


if __name__ == "__main__":
    main_with_validation()