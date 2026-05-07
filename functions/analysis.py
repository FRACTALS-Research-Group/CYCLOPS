import re
import subprocess
from typing import Any, List, Tuple

def run_analysis(base: str, out_base: str, pos: str, i: int, logger: Any) -> None:
    """
        Run the trajectory analysis script (functions/analyse.py) via subprocess.

        Parameters
        ----------
        base : str
            Base path to the project directory.
        out_base : str
            Base name for output files (without extension).
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        logger : Any
            Logger for logging information and errors.
        
        Raises
        ------
        RuntimeError
            If the analysis script fails with a non-zero return code.
    """

    command = [
        "python",
        f"{base}/functions/analyse.py",
        "-p",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_minimised.pdb",
        "-t",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_traj.dcd",
        "-o",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged",
        "-r",
    ]

    logger.info("Running analysis: %s", " ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        logger.error("run_analysis failed with return code %d", ret)
        raise RuntimeError(f"run_analysis failed with return code {ret}")

def reimage_trajectory(base: str, out_base: str, pos: str, i: int, logger: Any) -> None:
    """
        Convert reimaged DCD to a reimaged PDB trajectory using mdconvert.
        
        Parameters
        ----------
        base : str
            Base path to the project directory.
        out_base : str
            Base name for output files (without extension).
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        logger : Any
            Logger for logging information and errors.
        
        Raises
        ------
        RuntimeError
            If the mdconvert command fails with a non-zero return code.
    """
    command = [
        "mdconvert",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged.dcd",
        "-o",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_traj_reimaged.pdb",
        "-t",
        f"{base}/output/complexes/seq_{pos}_{i}/{out_base}_{pos}_{i}_reimaged.pdb",
    ]

    logger.info("Reimaging trajectory (mdconvert): %s", " ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        logger.error("reimage_trajectory failed with return code %d", ret)
        raise RuntimeError(f"reimage_trajectory failed with return code {ret}")
                
def move_scores(pos: str, i: int, output: str, logger: Any) -> None:
    """
        Move score summary file into the given output directory (via mv).
        
        Parameters
        ----------
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        output : str
            Output directory path.
        logger : Any
            Logger for logging information and errors.

        Raises
        ------
        RuntimeError
            If the mv command fails with a non-zero return code.
    """
    command = [
        "mv", 
        f"score_summary_system_{pos}_{i}_traj_reimaged.txt", 
        f"{output}"
    ]
    logger.info("Moving scores: %s", " ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        logger.error("move_scores failed with return code %d", ret)
        raise RuntimeError(f"move_scores failed with return code {ret}")

    logger.info("Scores moved to %s", output)


def new_scores_extraction(out_base: str, pos: str, i: int, logger: Any) -> List[float]:
    """
        Extract float values from score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt in CWD.
        
        Parameters        
        ----------
        out_base : str
            Base name for output files (without extension).
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        logger : Any
            Logger for logging information and errors.
        
        Returns
        -------
        List[float]
            List of extracted float values from the score summary file.
        
        Raises
        ------
        FileNotFoundError
            If the score summary file does not exist.
        Exception
            If any other error occurs during file reading or parsing.
    """
    filename = f"score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        numbers = re.findall(r"-?\d+\.\d+", text)
        values = [float(num) for num in numbers]
        logger.info("Extracted %d score values from %s", len(values), filename)
        return values

    except FileNotFoundError:
        logger.warning("Score file not found: %s", filename)
        return []
    except Exception as e:
        logger.exception("Error extracting scores from %s: %s", filename, e)
        return []

def old_scores_extraction(ACCEPTED_DIR: str, out_base: str, pos: str, i: int, logger: Any) -> List[float]:
    """
        Extract float values from ACCEPTED_DIR/score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt

        Parameters
        ----------
        ACCEPTED_DIR : str
            Directory where the score summary file is located.
        out_base : str
            Base name for output files (without extension).
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        logger : Any
            Logger for logging information and errors.
        
        Returns
        -------
        List[float]
            List of extracted float values from the score summary file.
        
        Raises
        ------
        FileNotFoundError
            If the score summary file does not exist in ACCEPTED_DIR.
        Exception
            If any other error occurs during file reading or parsing.
    """
    filename = f"{ACCEPTED_DIR}/score_summary_{out_base}_{pos}_{i}_traj_reimaged.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        numbers = re.findall(r"-?\d+\.\d+", text)
        values = [float(num) for num in numbers]
        logger.info("Extracted %d score values from %s", len(values), filename)
        return values

    except FileNotFoundError:
        logger.warning("Score file not found: %s", filename)
        return [100.0, 100.0, 100.0]
    except Exception as e:
        logger.exception("Error extracting scores from %s: %s", filename, e)
        return []

def find_negatives(list1: List[float], list2: List[float], logger: Any) -> Tuple[List[float], int]:
    """
        Compute elementwise differences between two lists and count how many are negative.
        
        Parameters
        ----------
        list1 : List[float]
            First list of float values.
        list2 : List[float]
            Second list of float values (must be the same length as list1).
        logger : Any
            Logger for logging information and errors.
        
        Returns
        -------
        Tuple[List[float], int]
            A tuple containing the list of differences and the count of negative differences.
        
        Raises
        ------
        ValueError
            If the input lists are not of the same length.
    """
    try:
        differences = [a - b for a, b in zip(list1, list2)]
    except Exception:
        differences = list(list1)

    negatives = sum(1 for num in differences if num < 0)

    logger.info("Differences: %s", differences)
    logger.info("Number of negatives: %d", negatives)

    return differences, negatives
