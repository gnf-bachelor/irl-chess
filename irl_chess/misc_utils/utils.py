def union_dicts(dict1, dict2):
    # Check for common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Error: Dictionaries have common keys: {common_keys}")

    # If no common keys, perform the union
    return {**dict1, **dict2}

def reformat_list(lst, inbetween_char = ''):
    return inbetween_char.join(map(str, lst))