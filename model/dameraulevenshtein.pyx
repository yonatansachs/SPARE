import jellyfish
import numpy as np


def damerau_levenshtein_distance(seq1, seq2):
    """
    Compute the Damerau-Levenshtein distance using the jellyfish library.

    Args:
        seq1 (str): The first string or sequence.
        seq2 (str): The second string or sequence.

    Returns:
        int: Damerau-Levenshtein distance between seq1 and seq2.

    Examples:
        >>> damerau_levenshtein_distance('smtih', 'smith')
        1
        >>> damerau_levenshtein_distance('saturday', 'sunday')
        3
        >>> damerau_levenshtein_distance('orange', 'pumpkin')
        7
    """
    return jellyfish.damerau_levenshtein_distance(seq1, seq2)


def normalized_damerau_levenshtein_distance(seq1, seq2):
    """
    Compute the normalized Damerau-Levenshtein distance as a fraction of the longer string.

    Args:
        seq1 (str): The first string or sequence.
        seq2 (str): The second string or sequence.

    Returns:
        float: Normalized Damerau-Levenshtein distance (0.0 to 1.0).

    Examples:
        >>> normalized_damerau_levenshtein_distance('smtih', 'smith')
        0.2
        >>> normalized_damerau_levenshtein_distance('saturday', 'sunday')
        0.375
        >>> normalized_damerau_levenshtein_distance('orange', 'pumpkin')
        1.0
    """
    n = max(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    return damerau_levenshtein_distance(seq1, seq2) / n


def damerau_levenshtein_distance_ndarray(seq, array):
    """
    Compute the Damerau-Levenshtein distance for each element in an array relative to a given sequence.

    Args:
        seq (str): The sequence to compare against.
        array (list or np.ndarray): An array of sequences to compare.

    Returns:
        np.ndarray: Array of Damerau-Levenshtein distances.

    Examples:
        >>> damerau_levenshtein_distance_ndarray('smith', ['smtih', 'saturday', 'pumpkin'])
        array([1, 6, 7], dtype=uint32)
    """
    dl = np.vectorize(damerau_levenshtein_distance, otypes=[np.uint32])
    return dl(seq, array)


def normalized_damerau_levenshtein_distance_ndarray(seq, array):
    """
    Compute the normalized Damerau-Levenshtein distance for each element in an array relative to a given sequence.

    Args:
        seq (str): The sequence to compare against.
        array (list or np.ndarray): An array of sequences to compare.

    Returns:
        np.ndarray: Array of normalized Damerau-Levenshtein distances.

    Examples:
        >>> normalized_damerau_levenshtein_distance_ndarray('smith', ['smtih', 'saturday', 'pumpkin'])
        array([0.2, 0.8571428571428571, 1.0], dtype=float32)
    """
    ndl = np.vectorize(normalized_damerau_levenshtein_distance, otypes=[np.float32])
    return ndl(seq, array)


def damerau_levenshtein_diversity(array):
    """
    Compute the average pairwise Damerau-Levenshtein distance among all elements in an array.

    Args:
        array (list or np.ndarray): An array of sequences.

    Returns:
        float: Average pairwise Damerau-Levenshtein distance.

    Examples:
        >>> damerau_levenshtein_diversity(['smith', 'saturday', 'pumpkin'])
        7.666666666666667
    """
    dl_distance = 0
    len_array = len(array)
    if len_array < 2:
        return 0.0

    for i in range(len_array):
        for j in range(i + 1, len_array):
            dl_distance += damerau_levenshtein_distance(array[i], array[j])

    return dl_distance / (len_array * (len_array-1/2))