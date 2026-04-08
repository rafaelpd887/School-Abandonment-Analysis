import nbformat as nbf

# Path to the original script
script_path = "scripts/eda.py"

# Name of the notebook to be created
notebook_path = "notebooks/eda_notebook.ipynb"

# Read the script content
with open(script_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Create a new notebook
nb = nbf.v4.new_notebook()

# Each block of code separated by blank lines becomes a cell
cell_lines = []
for line in lines:
    if line.strip() == "":  # blank line = new cell
        if cell_lines:
            nb.cells.append(nbf.v4.new_code_cell("".join(cell_lines)))
            cell_lines = []
    else:
        cell_lines.append(line)

# Add the last cell if anything is left
if cell_lines:
    nb.cells.append(nbf.v4.new_code_cell("".join(cell_lines)))

# Save the notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created at: {notebook_path}")