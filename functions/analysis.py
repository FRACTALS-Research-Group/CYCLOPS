import subprocess

def run_analysis(base, out_base, pos, i):
    command = ['python', f'{base}/functions/analyse.py', '-p', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_minimised.pdb', '-t', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_traj.dcd', '-o', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged', '-r']
    subprocess.call(command)
                
def reimage_trajectory(base, out_base, pos, i):
    command = ['mdconvert', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged.dcd', '-o', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_traj_reimaged.pdb', '-t', f'{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged.pdb']
    subprocess.call(command)

def move_scores(pos, i, output):
    command = ['mv', f'score_summary_system_{pos}_{i}_traj_reimaged.txt', f'{output}']
    subprocess.call(command)
    print(f"Scores moved in {output}")
    
import re

def new_scores_extraction(out_base, pos, i):
    try:
        # Apri il file in modalità lettura
        with open(f"score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt", 'r') as file:
            # Leggi il contenuto del file
            text = file.read()

        # Usa una regex per trovare tutti i numeri
        numbers = re.findall(r"-?\d+\.\d+", text)

        # Converti i numeri in formato float
        return [float(num) for num in numbers]
    
    except FileNotFoundError:
        print(f"Error: The file score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt does not exist.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def old_scores_extraction(ACCEPTED_DIR, out_base, pos, i):
    try:
        # Apri il file in modalità lettura
        with open(f"{ACCEPTED_DIR}/score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt", 'r') as file:
            # Leggi il contenuto del file
            text = file.read()

        # Usa una regex per trovare tutti i numeri
        numbers = re.findall(r"-?\d+\.\d+", text)

        # Converti i numeri in formato float
        return [float(num) for num in numbers]
    
    except FileNotFoundError:
        print(f"Error: The file score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt does not exist.")
        return [100, 100, 100]
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def find_negatives(lista1, lista2):
    # if len(lista1) != len(lista2):
    #     print("Error: The lists must be the same length")
    #     return []

    # Sottrae i numeri corrispondenti
    try:
        differences = [a - b for a, b in zip(lista1, lista2)]
        print(f"Differences: {differences}")
    except: differences = lista1
    
    try:
        negatives = len([num for num in differences if num < 0])
        print(f"Number of negatives: {negatives}")
    except: negatives = 0
    
    return differences, negatives
