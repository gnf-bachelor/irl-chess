import scipy
import numpy as np


def wilson_score_interval(successes, total, confidence=0.95):
    p_hat = successes / total
    z = np.sqrt(2) * scipy.special.erfinv(confidence)
    denominator = 1 + z**2 / total
    center_adjusted_probability = p_hat + z**2 / (2 * total)
    adjusted_standard_deviation = np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    lower_bound = (center_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (center_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound
