#!/usr/bin/env python3
"""
MMMU Data Collector for multimodal data collection
Handles MMMU dataset loading for data collection purposes
"""

import os
import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from .data_utils import process_single_sample, construct_prompt, DOMAIN_CAT2SUB_CAT


class MMMUDataCollector:
    """MMMU data collector for extracting samples without model inference"""
    
    def __init__(self, dataset_path, split="test", subset="Medical"):
        """
        Initialize MMMU data collector
        
        Args:
            dataset_path: Path to MMMU dataset
            split: Dataset split ("test" or "validation")
            subset: Dataset subset ("Medical" or "Science")
        """
        self.dataset_path = dataset_path
        self.split = split
        self.subset = subset
        
        if self.subset == "Medical":
            self.subset_long = "Health and Medicine"
        elif self.subset == "Science":
            self.subset_long = "Science"
        else:
            raise ValueError(f"Unsupported subset: {subset}")
    
    def load_data(self):
        """
        Load MMMU data and extract samples
        
        Returns:
            List of sample dictionaries
        """
        sub_dataset_list = []
        
        # Load all subjects in the specified subset
        for subject in tqdm(DOMAIN_CAT2SUB_CAT[self.subset_long], desc=f"Loading {self.subset} subjects"):
            try:
                sub_dataset = load_dataset(self.dataset_path, subject, split=self.split)
                sub_dataset_list.append(sub_dataset)
            except Exception as e:
                print(f"Warning: Failed to load subject {subject}: {e}")
                continue
        
        if not sub_dataset_list:
            raise ValueError(f"No subjects found for subset {self.subset_long}")
        
        # Concatenate all subjects
        dataset = concatenate_datasets(sub_dataset_list)
        
        # Process samples
        samples = []
        for idx, sample in tqdm(enumerate(dataset), desc="Processing MMMU samples"):
            try:
                # Process single sample
                processed_sample = process_single_sample(sample)
                
                # Construct prompt
                processed_sample = construct_prompt(processed_sample)
                
                # Add dataset identifier
                processed_sample["dataset"] = f"MMMU-{self.subset}-{self.split}"
                
                samples.append(processed_sample)
            except Exception as e:
                print(f"Warning: Failed to process sample {idx}: {e}")
                continue
        
        return samples


def load_mmmu_data(dataset_path, split="test", subset="Medical"):
    """
    Convenience function to load MMMU data
    
    Args:
        dataset_path: Path to MMMU dataset
        split: Dataset split ("test" or "validation")
        subset: Dataset subset ("Medical" or "Science")
    
    Returns:
        List of sample dictionaries
    """
    collector = MMMUDataCollector(dataset_path, split, subset)
    return collector.load_data()