# Converted from Untitled8.ipynb

import os

def find_large_files(directory, top_n=50, min_size_mb=1):
    """
    Find the largest files in a directory.
    
    Args:
        directory: Path to search
        top_n: Number of largest files to show
        min_size_mb: Minimum file size in MB to include
    """
    
    file_sizes = []
    
    print(f"Scanning {directory}...")
    
    for root, dirs, files in os.walk(directory):
        # Skip common large/irrelevant directories
        dirs[:] = [d for d in dirs if d not in [
            'node_modules', '.git', '__pycache__', '.conda', 
            'venv', 'env', '.venv', 'site-packages'
        ]]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
                if size >= min_size_mb * 1024 * 1024:  # Convert MB to bytes
                    file_sizes.append((filepath, size))
            except (OSError, PermissionError):
                continue
    
    # Sort by size descending
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} LARGEST FILES IN {directory}")
    print(f"{'='*80}\n")
    
    print(f"{'Size (MB)':<12} {'File Path'}")
    print("-"*80)
    
    total_size = 0
    for filepath, size in file_sizes[:top_n]:
        size_mb = size / (1024 * 1024)
        total_size += size
        # Show path relative to search directory
        rel_path = os.path.relpath(filepath, directory)
        print(f"{size_mb:>10.1f}  {rel_path}")
    
    print("-"*80)
    print(f"{'Total:':<12} {total_size / (1024**3):.2f} GB in top {min(top_n, len(file_sizes))} files")
    print(f"{'All files:':<12} {sum(s for _, s in file_sizes) / (1024**3):.2f} GB ({len(file_sizes)} files >= {min_size_mb}MB)")
    
    return file_sizes


# Run it on your code folder
large_files = find_large_files(r'C:\mozg\code', top_n=30, min_size_mb=10)
