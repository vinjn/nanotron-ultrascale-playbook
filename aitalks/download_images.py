#!/usr/bin/env python3
"""
Script to scan markdown files, download images from hyperlinks, 
and convert hyperlinks to reference local images.
"""

import os
import re
import requests
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin
import argparse
import sys
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from functools import partial
import threading

class ImageDownloader:
    def __init__(self, base_dir: str, images_dir: str = "images", max_workers: int = None):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / images_dir
        self.images_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers or min(8, cpu_count())
        
        # Pattern to match markdown image syntax: ![alt](url)
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'}
        
        # Thread-local storage for session to avoid sharing across processes
        self._local = threading.local()

    @property
    def session(self):
        """Get thread-local session to avoid sharing across processes"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
        return self._local.session

    def is_url(self, path: str) -> bool:
        """Check if the path is a URL (http/https)"""
        return path.startswith(('http://', 'https://'))

    def is_image_url(self, url: str) -> bool:
        """Check if URL points to an image based on extension or content type"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check file extension
        for ext in self.image_extensions:
            if path.endswith(ext):
                return True
        
        # If no clear extension, try to check content type
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return content_type.startswith('image/')
        except:
            return False

    def get_image_extension(self, url: str, content_type: str = None) -> str:
        """Get appropriate file extension for the image"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Try to get extension from URL
        for ext in self.image_extensions:
            if path.endswith(ext):
                return ext
        
        # Try to get extension from content type
        if content_type:
            content_type = content_type.lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                return '.jpg'
            elif 'png' in content_type:
                return '.png'
            elif 'gif' in content_type:
                return '.gif'
            elif 'svg' in content_type:
                return '.svg'
            elif 'webp' in content_type:
                return '.webp'
        
        # Default to .jpg if we can't determine
        return '.jpg'

    def generate_filename(self, url: str, content_type: str = None) -> str:
        """Generate a unique filename for the image"""
        # Create a hash of the URL for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        
        # Get the original filename if available
        parsed = urlparse(url)
        original_name = Path(parsed.path).stem
        
        if original_name and len(original_name) > 0:
            # Clean the filename
            original_name = re.sub(r'[^\w\-_.]', '_', original_name)[:50]
            filename = f"{original_name}_{url_hash}"
        else:
            filename = f"image_{url_hash}"
        
        # Add appropriate extension
        extension = self.get_image_extension(url, content_type)
        return f"{filename}{extension}"

    def download_image(self, url: str) -> str:
        """Download an image and return the local filename"""
        try:
            print(f"Downloading: {url}")
            
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            filename = self.generate_filename(url, content_type)
            filepath = self.images_dir / filename
            
            # Check if file already exists
            if filepath.exists():
                print(f"  File already exists: {filename}")
                return filename
            
            # Write the image data
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"  Saved as: {filename}")
            return filename
            
        except Exception as e:
            print(f"  Error downloading {url}: {str(e)}")
            return None

    def process_markdown_file(self, file_path: Path) -> bool:
        """Process a single markdown file and download images"""
        try:
            print(f"\nProcessing: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modifications = []
            
            # Process markdown image syntax: ![alt](url)
            for match in self.image_pattern.finditer(content):
                alt_text = match.group(1)
                url = match.group(2)
                
                if self.is_url(url) and self.is_image_url(url):
                    local_filename = self.download_image(url)
                    if local_filename:
                        # Calculate relative path from markdown file to image
                        relative_path = os.path.relpath(self.images_dir / local_filename, file_path.parent)
                        # Use forward slashes for markdown
                        relative_path = relative_path.replace('\\', '/')
                        
                        new_link = f"![{alt_text}]({relative_path})"
                        modifications.append((match.group(0), new_link))
            
            # Apply modifications
            if modifications:
                for old_text, new_text in modifications:
                    content = content.replace(old_text, new_text)
                
                # Write back the modified content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  Updated {len(modifications)} image references")
                return True
            else:
                print(f"  No remote images found")
                return False
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def scan_directory(self, directory: Path = None, use_multiprocessing: bool = True) -> Dict[str, int]:
        """Scan directory for markdown files and process them"""
        if directory is None:
            directory = self.base_dir
        
        stats = {"files_processed": 0, "files_modified": 0, "images_downloaded": 0}
        
        # Find all markdown files
        markdown_files = []
        for pattern in ['*.md', '*.markdown']:
            markdown_files.extend(directory.rglob(pattern))
        
        if not markdown_files:
            print(f"No markdown files found in {directory}")
            return stats
        
        print(f"Found {len(markdown_files)} markdown files")
        
        images_before = len(list(self.images_dir.glob('*'))) if self.images_dir.exists() else 0
        
        if use_multiprocessing and len(markdown_files) > 1:
            print(f"Using {self.max_workers} worker processes")
            
            # Create a partial function with the downloader instance
            process_func = partial(process_markdown_file_worker, 
                                 base_dir=str(self.base_dir),
                                 images_dir=str(self.images_dir.relative_to(self.base_dir)))
            
            with Pool(processes=self.max_workers) as pool:
                results = pool.map(process_func, markdown_files)
            
            # Aggregate results
            for result in results:
                stats["files_processed"] += 1
                if result:
                    stats["files_modified"] += 1
        else:
            # Sequential processing
            for md_file in markdown_files:
                stats["files_processed"] += 1
                if self.process_markdown_file(md_file):
                    stats["files_modified"] += 1
        
        images_after = len(list(self.images_dir.glob('*'))) if self.images_dir.exists() else 0
        stats["images_downloaded"] = images_after - images_before
        
        return stats

def process_markdown_file_worker(file_path: Path, base_dir: str, images_dir: str) -> bool:
    """Worker function for multiprocessing - processes a single markdown file"""
    try:
        # Create a new ImageDownloader instance for this worker
        downloader = ImageDownloader(base_dir, images_dir)
        return downloader.process_markdown_file(file_path)
    except Exception as e:
        print(f"Error in worker processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download images from markdown files and convert links to local references')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to scan for markdown files (default: current directory)')
    parser.add_argument('--images-dir', default='images', help='Directory to store downloaded images (default: images)')
    parser.add_argument('--workers', type=int, help='Number of worker processes (default: min(8, CPU count))')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing and run sequentially')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    base_dir = Path(args.directory).resolve()
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)
    
    print(f"Scanning directory: {base_dir}")
    print(f"Images will be stored in: {base_dir / args.images_dir}")
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run mode
        return
    
    downloader = ImageDownloader(base_dir, args.images_dir, args.workers)
    use_mp = not args.no_multiprocessing
    stats = downloader.scan_directory(use_multiprocessing=use_mp)
    
    print(f"\n=== Summary ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files modified: {stats['files_modified']}")
    print(f"Images downloaded: {stats['images_downloaded']}")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
