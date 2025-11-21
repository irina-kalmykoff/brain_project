# convert_notebooks_to_py.py
import os
import json
import sys

def convert_notebook_to_py(notebook_path, py_path):
    """Convert a single notebook to a Python file."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    py_content = []
    
    # Add header
    py_content.append(f"# Converted from {os.path.basename(notebook_path)}")
    py_content.append("")
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            # Add code cells
            source = ''.join(cell['source'])
            if source.strip():  # Only add non-empty cells
                py_content.append(source)
                py_content.append("")
                
        elif cell['cell_type'] == 'markdown':
            # Add markdown cells as comments
            source = ''.join(cell['source'])
            if source.strip():
                # Convert each line to a comment
                for line in source.split('\n'):
                    py_content.append(f"# {line}")
                py_content.append("")
    
    # Write to Python file
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(py_content))
    
    print(f"Converted: {os.path.basename(notebook_path)} -> {os.path.basename(py_path)}")

def convert_all_notebooks(directory='.'):
    """Convert all notebooks in a directory to Python files."""
    
    # Find all .ipynb files
    notebook_files = [f for f in os.listdir(directory) 
                     if f.endswith('.ipynb') and not f.startswith('.ipynb_checkpoints')]
    
    if not notebook_files:
        print("No notebook files found in the current directory.")
        return
    
    print(f"Found {len(notebook_files)} notebook(s) to convert:")
    for nb in notebook_files:
        print(f"  - {nb}")
    
    print("\nConverting...")
    
    converted_count = 0
    error_count = 0
    
    for notebook_file in notebook_files:
        notebook_path = os.path.join(directory, notebook_file)
        py_file = notebook_file.replace('.ipynb', '.py')
        py_path = os.path.join(directory, py_file)
        
        # Check if .py file already exists
        if os.path.exists(py_path):
            print(f"Warning: {py_file} exists and will be overwritten")
        
        try:
            convert_notebook_to_py(notebook_path, py_path)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {notebook_file}: {e}")
            error_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Successfully converted: {converted_count}")
    if error_count > 0:
        print(f"  Errors: {error_count}")

if __name__ == "__main__":
    # Use command line argument for directory if provided
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    # Get absolute path for clarity
    abs_directory = os.path.abspath(directory)
    print(f"Converting notebooks in: {abs_directory}\n")
    
    convert_all_notebooks(directory)