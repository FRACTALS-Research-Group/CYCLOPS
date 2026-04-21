import random

def mutation(sequence, position, amino_acids):
    """
    Sostituisce un amminoacido in una sequenza con uno casuale dalla lista di amminoacidi.

    Args:
        sequence (str): La sequenza originale di amminoacidi.
        position (int): La posizione dell'amminoacido da sostituire (0-indice).
        amino_acids (list): Lista di amminoacidi possibili.

    Returns:
        str: La sequenza mutata con l'amminoacido sostituito.
    """
    if position < 0 or position >= len(sequence):
        raise ValueError("La posizione deve essere valida per la sequenza.")

    # Scegli un amminoacido casuale dalla lista
    new_amino_acid = random.choice(amino_acids)

    # Crea una nuova sequenza con l'amminoacido mutato
    mutated_sequence = sequence[:position] + new_amino_acid + sequence[position + 1:]

    return mutated_sequence

def random_n_mutation(sequence, valid_positions, amino_acids, n=2):
    """
    Sostituisce 'n' amminoacidi in posizioni casuali con altri casuali dalla lista di amminoacidi,
    assicurandosi che il nuovo amminoacido sia diverso da quello originale.

    Args:
        sequence (str): La sequenza originale di amminoacidi.
        amino_acids (list): Lista di amminoacidi possibili.
        n (int): Numero di mutazioni da applicare.

    Returns:
        str: La sequenza mutata.
    """
    if n > len(valid_positions):
        raise ValueError("Il numero di mutazioni supera le posizioni mutabili disponibili.")

    sequence = list(sequence)
    positions = random.sample(valid_positions, n)
    
    for pos in positions:
        original = sequence[pos]
        choices = [aa for aa in amino_acids if aa != original]
        sequence[pos] = random.choice(choices)
    
    mutate_seq = ''.join(sequence)
    return mutate_seq, positions


import random

def random_start_seq():
    amminoacidi = "ACDEFGHIKLMNPQRSTVWY"
    stringa_casuale = "C" + ''.join(random.choices(amminoacidi, k=10)) + "C"
    return stringa_casuale
