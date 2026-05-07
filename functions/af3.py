#
# Copyright (c) 2026 FRACTALS Research Group
# Visit the Research Group website for more information: https://fractals.group/
#

"""
    AlphaFold3 runner utility.
"""
import subprocess

DEFAULT_AF3_SCRIPT = "/opt/miniconda3/envs/alphafold3/alphafold3/run_alphafold.py"

def alphafold3(
        json_path: str, 
        output_dir: str, 
        logger
    ) -> subprocess.CompletedProcess:
    """
        Runs the AlphaFold3 script with the specified JSON input and output directory.
        
        Parameters
        ----------
        json_path : str
            Path to the JSON file containing the input data for AlphaFold3.
        output_dir : str
            Directory where AlphaFold3 will save its output.
        logger : logging.Logger
            Logger for logging information and errors.
        
        Returns
        -------
        subprocess.CompletedProcess
            The result of the subprocess execution, containing return code, stdout, and stderr.
        
        Raises
        ------
        RuntimeError
            If the AlphaFold3 script fails.
    """
    command = [
        "python",
        DEFAULT_AF3_SCRIPT,
        "--json_path", str(json_path),
        "--output_dir", str(output_dir),
    ]

    logger.info("Running AlphaFold3: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    if result.stdout:
        logger.info("AlphaFold3 stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("AlphaFold3 stderr:\n%s", result.stderr)

    if result.returncode != 0:
        logger.error("AlphaFold3 failed with return code %s", result.returncode)
        raise RuntimeError(f"AlphaFold3 failed with return code {result.returncode}")

    logger.info("AlphaFold3 finished successfully.")
    return result