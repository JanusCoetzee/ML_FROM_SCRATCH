import numpy as np


def convert_to_binary(data, value_for_zero, value_for_one):
    """
    Convert an array with two distinct values to binary (0 and 1).
    
    Parameters:
    -----------
    data : array-like
        Input array with two distinct values
    value_for_zero : any
        The value that should be mapped to 0
    value_for_one : any
        The value that should be mapped to 1
        
    Returns:
    --------
    numpy.ndarray
        Binary array with 0s and 1s
        
    Example:
    --------
    >>> convert_to_binary([True, False, True, False], False, True)
    array([1, 0, 1, 0])
    >>> convert_to_binary(['yes', 'no', 'yes'], 'no', 'yes')
    array([1, 0, 1])
    """
    data = np.array(data)
    binary_array = np.where(data == value_for_zero, 0, 
                           np.where(data == value_for_one, 1, np.nan))
    
    # Check for unexpected values
    if np.any(np.isnan(binary_array)):
        raise ValueError(f"Data contains values other than {value_for_zero} and {value_for_one}")
    
    return binary_array.astype(int)
