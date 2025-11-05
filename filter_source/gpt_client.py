"""
GPT Client for Medical Data Filtering

This module provides a GPT client for medical data filtering using OpenAI-compatible APIs.
It supports image and text processing with configurable API endpoints and authentication.

Key Features:
- OpenAI-compatible API support
- Image compression and processing
- Configurable API endpoints and authentication
- Error handling and retry logic
- Support for multiple image formats
"""

import io
import base64
import time
from typing import List, Optional, Dict, Any
from PIL import Image


class GPT5V:
    """GPT client for judgment using OpenAI-compatible APIs"""
    
    def __init__(self, api_key: str, api_url: str, model_name='gpt-4',
                 max_completion_tokens=4096, reasoning_effort='medium',
                 verbosity='medium', max_image_dimension=1024,
                 compress_on_413_only=True):
        """
        Initialize the GPT client
        
        Args:
            api_key: API key for authentication
            api_url: Base URL for the API endpoint
            model_name: GPT model name
            max_completion_tokens: Maximum tokens for completion
            reasoning_effort: Reasoning effort level
            verbosity: Verbosity level
            max_image_dimension: Maximum image dimension for compression
            compress_on_413_only: Whether to compress images only on 413 errors
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_image_dimension = max_image_dimension
        self.compress_on_413_only = compress_on_413_only
        self.compress_attempts = 0  # Track compression attempts for reporting
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )
        except ImportError:
            print("❌ ERROR: OpenAI library not installed. Install with: pip install openai")
            raise

    def _compress_image(self, img, max_dimension=None, quality=100, verbose=False, compression_type="initial"):
        """
        Compress image by controlling pixel count and/or quality
        Used only for GPT upload to prevent 413 errors
        
        Args:
            img: PIL Image object
            max_dimension: Maximum dimension (width or height), None uses class default
            quality: JPEG quality (1-100)
            verbose: Whether to report compression details
            compression_type: Type of compression for reporting ("initial", "aggressive")
            
        Returns:
            Resized PIL Image object
        """
        if max_dimension is None:
            max_dimension = self.max_image_dimension
            
        original_size = img.size
        original_mode = img.mode
        
        # Only resize if image is too large
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            if verbose:
                print(f"    📏 IMAGE COMPRESSION ({compression_type}): Resized from {original_size} to {new_size}")
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ['P', 'RGBA', 'LA', 'PA']:
            if 'A' in img.mode:
                # For images with alpha channel, create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img, mask=img)
                img = background
                if verbose:
                    print(f"    🎨 IMAGE COMPRESSION ({compression_type}): Converted from {original_mode} to RGB (with alpha removal)")
            else:
                # For palette images, convert directly
                img = img.convert('RGB')
                if verbose:
                    print(f"    🎨 IMAGE COMPRESSION ({compression_type}): Converted from {original_mode} to RGB")
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            if verbose:
                print(f"    🎨 IMAGE COMPRESSION ({compression_type}): Converted from {original_mode} to RGB")
        
        self.compress_attempts += 1
        return img, quality

    def call(self, content, images=None, args={}, verbose=False, max_retries=3, dry_run=False):
        """
        Call GPT API with text and optional images using OpenAI library
        
        Args:
            content: Text content
            images: List of PIL Image objects or single PIL Image
            args: Additional parameters
            verbose: Whether to print verbose output
            max_retries: Maximum number of retry attempts
            dry_run: Whether in dry run mode
            
        Returns:
            GPT response string
        """
        # Prepare messages
        messages = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        
        # Handle images
        if images:
            if not isinstance(images, list):
                images = [images]
                
            for img in images:
                if hasattr(img, 'size'):  # PIL Image object
                    # Create a copy for compression (don't modify original)
                    img_copy = img.copy()
                    
                    # Report original image details in dry_run mode
                    if dry_run or verbose:
                        original_size = img_copy.size
                        original_mode = img_copy.mode
                        print(f"    📊 IMAGE INFO: Original size {original_size}, mode {original_mode}")
                    
                    # Determine compression strategy
                    if self.compress_on_413_only:
                        # First attempt: no compression, only format conversion if needed
                        img_processed, quality = self._compress_image(
                            img_copy,
                            max_dimension=None,  # No resizing on first attempt
                            quality=100,
                            verbose=(dry_run or verbose),
                            compression_type="format_only"
                        )
                    else:
                        # Always compress: resize if too large
                        img_processed, quality = self._compress_image(
                            img_copy,
                            max_dimension=self.max_image_dimension,
                            quality=100,
                            verbose=(dry_run or verbose),
                            compression_type="initial"
                        )
                    
                    # Convert processed PIL Image to base64
                    img_bytes = io.BytesIO()
                    img_processed.save(img_bytes, format='JPEG', quality=quality, optimize=True)
                    img_bytes.seek(0)
                    image_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                    
                    # Report image details
                    if dry_run or verbose:
                        size_kb = len(img_bytes.getvalue()) / 1024
                        print(f"    📄 IMAGE UPLOAD: Base64 length {len(image_base64)}, file size {size_kb:.1f}KB, quality {quality}%")
                    
                    # Add image to messages
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Print verbose request details
        if verbose:
            print(f"\n=== GPT API REQUEST DETAILS ===")
            print(f"Model: {self.model_name}")
            print(f"Content: {content[:200]}...")
            print(f"Images: {len(images) if images else 0}")
            
            # Create a copy of messages without base64 image data for cleaner output
            clean_messages = []
            for message in messages:
                clean_message = message.copy()
                if 'content' in clean_message and isinstance(clean_message['content'], list):
                    clean_content = []
                    for content_item in clean_message['content']:
                        if content_item.get('type') == 'image_url' and 'image_url' in content_item:
                            clean_content.append({
                                "type": "image_url",
                                "image_url": {"url": "[BASE64_IMAGE_DATA]"}
                            })
                        else:
                            clean_content.append(content_item)
                    clean_message['content'] = clean_content
                clean_messages.append(clean_message)
            
            print(f"Messages: {clean_messages}")
            print(f"=== END REQUEST DETAILS ===\n")
        
        # Retry logic with improved error handling
        for attempt in range(max_retries):
            try:
                # Call GPT with image using OpenAI library
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_completion_tokens,
                    **args
                )
                
                gpt_response = response.choices[0].message.content
                
                # Print verbose response details
                if verbose:
                    print(f"\n=== GPT API RESPONSE DETAILS ===")
                    print(f"Response: {gpt_response}")
                    print(f"=== END RESPONSE DETAILS ===\n")
                
                return gpt_response
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific error types
                if "413" in error_msg:
                    print(f"❌ GPT API error (attempt {attempt + 1}/{max_retries}): Request too large (413)")
                    if attempt < max_retries - 1:
                        # Try with more aggressive compression
                        print("    🔄 Retrying with more aggressive image compression...")
                        
                        # Re-process images with compression
                        if images and self.compress_on_413_only:
                            # Update messages with compressed images
                            messages[0]["content"] = [{"type": "text", "text": content}]
                            for img in images:
                                if hasattr(img, 'size'):
                                    img_copy = img.copy()
                                    # Apply aggressive compression
                                    img_compressed, quality = self._compress_image(
                                        img_copy,
                                        max_dimension=self.max_image_dimension,
                                        quality=85,  # Lower quality for aggressive compression
                                        verbose=(dry_run or verbose),
                                        compression_type="aggressive_413"
                                    )
                                    
                                    # Convert to base64
                                    img_bytes = io.BytesIO()
                                    img_compressed.save(img_bytes, format='JPEG', quality=quality, optimize=True)
                                    img_bytes.seek(0)
                                    image_base64 = base64.b64encode(img_bytes.getvalue()).decode()
                                    
                                    if dry_run or verbose:
                                        size_kb = len(img_bytes.getvalue()) / 1024
                                        print(f"    🔄 RETRY IMAGE: Aggressive compression to {size_kb:.1f}KB, quality {quality}%")
                                    
                                    messages[0]["content"].append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                                    })
                        continue
                    else:
                        print("❌ Failed after multiple attempts with image compression")
                        return "no"  # Fallback to "no" for judgment
                elif "429" in error_msg:
                    print(f"❌ GPT API error (attempt {attempt + 1}/{max_retries}): Rate limit exceeded (429)")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ Rate limit persists after multiple attempts")
                        return "no"  # Fallback to "no" for judgment
                else:
                    print(f"❌ GPT API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print(f"❌ Failed after {max_retries} attempts")
                        return "no"  # Fallback to "no" for judgment