import subprocess
def alphafold3(json_path, output_dir):
    command = [
    "python", "/opt/miniconda3/envs/alphafold3/alphafold3/run_alphafold.py", "--json_path", json_path, "--output_dir", output_dir]
    
    result = subprocess.run(command, capture_output=True, text=True)
    print("Standard Output:\n", result.stdout)
    print("Standard Error:\n", result.stderr)