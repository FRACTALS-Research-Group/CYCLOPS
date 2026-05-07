import random
from typing import List, Sequence, Tuple


def mutation(sequence: str, position: int, amino_acids: Sequence[str]) -> str:
    """
        Substitute a single amino acid in a sequence at a given position.

        Parameters
        ----------
        sequence : str
            Original amino-acid sequence.
        position : int
            Position to mutate (0-based).
        amino_acids : Sequence[str]
            Allowed amino acids to sample from.

        Returns
        -------
        str
            Mutated sequence.

        Raises
        ------
        ValueError
            If position is out of range [0, len(sequence)-1].
    """
    if position < 0 or position >= len(sequence):
        raise ValueError("Position must be within the sequence length.")

    new_amino_acid = random.choice(list(amino_acids))
    return sequence[:position] + new_amino_acid + sequence[position + 1:]

def random_n_mutation(sequence: str, valid_positions: Sequence[int], 
                      amino_acids: Sequence[str], n: int = 2
                    ) -> Tuple[str, List[int]]:
    """
        Apply n random mutations at distinct positions selected from valid_positions,
        ensuring the new amino acid differs from the original at each position.

        Parameters
        ----------
        sequence : str
            Original amino-acid sequence.
        valid_positions : Sequence[int]
            Positions eligible for mutation (0-based).
        amino_acids : Sequence[str]
            Allowed amino acids to sample from.
        n : int
            Number of mutations.

        Returns
        -------
        (mutated_sequence, positions) : (str, List[int])
            Mutated sequence and the list of mutated positions.

        Raises
        ------
        ValueError
            If n exceeds the number of valid positions or if amino_acids is invalid.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0 (got {n}).")

    if n > len(valid_positions):
        raise ValueError("Number of mutations exceeds available mutable positions.")

    aa_list = list(amino_acids)
    if len(aa_list) < 2:
        raise ValueError("amino_acids must contain at least 2 entries to enforce aa != original.")

    seq_list = list(sequence)
    positions = random.sample(list(valid_positions), n)

    for pos in positions:
        original = seq_list[pos]
        choices = [aa for aa in aa_list if aa != original]
        seq_list[pos] = random.choice(choices)

    mutate_seq = "".join(seq_list)
    return mutate_seq, positions

def random_start_seq(k: int = 10) -> str:
    """
        Generate a random starting peptide sequence: 'C' + k random amino acids + 'C'.

        Parameters
        ----------
        k : int
            Length of the random middle segment.

        Returns
        -------
        str
            Random sequence.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "C" + "".join(random.choices(amino_acids, k=k)) + "C"

