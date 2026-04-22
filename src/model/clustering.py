import numpy as np
import json
from pathlib import Path

def subtractive_cluster(
    data: np.ndarray,
    radius: float = 0.5,
    squash: float = 1.25,
    accept_ratio: float = 0.5,
    reject_ratio: float = 0.15
) -> np.ndarray:
    """Implements Chiu's Subtractive Clustering to find fuzzy rule centers.

    Args:
        data: The feature matrix to cluster (shape: n_samples, n_features).
        radius: The defining radius of a cluster.
        squash: Multiplier for the radius when subtracting potential to prevent 
            closely spaced clusters.
        accept_ratio: Threshold fraction of max potential to definitively accept a center.
        reject_ratio: Threshold fraction of max potential to definitively reject a center.

    Returns:
        An array of cluster centers (shape: n_clusters, n_features).

    Raises:
        ValueError: If the dataset contains fewer than 2 rows.
    """
    if data.shape[0] < 2:
        raise ValueError("Data must have at least 2 rows to perform clustering.")

    # Calculate the mathematical constants for the potential function
    # The multiplier 4.0 is standard in Chiu's formula for the exponential decay
    radius_sq = (radius / 2.0) ** 2
    alpha = 4.0 / radius_sq
    
    squash_sq = (radius * squash / 2.0) ** 2
    beta = 4.0 / squash_sq

    n_samples = data.shape[0]
    potentials = np.zeros(n_samples)

    # 1. Calculate Initial Potentials for all data points
    # This measures how many other points are nearby. High potential = dense area.
    for i in range(n_samples):
        # We calculate the squared Euclidean distance from point i to all other points
        diff = data - data[i]
        dist_sq = np.sum(diff ** 2, axis=1)
        potentials[i] = np.sum(np.exp(-alpha * dist_sq))

    # Store the maximum potential found in the first pass
    highest_potential = np.max(potentials)
    max_idx = np.argmax(potentials)
    first_max_potential = highest_potential
    
    cluster_centres = []

    # 2. Iteratively extract cluster centers
    while highest_potential > 0:
        current_centre = data[max_idx]
        potential_ratio = highest_potential / first_max_potential

        accept = False

        # Condition A: Upper bound acceptance
        if potential_ratio > accept_ratio:
            accept = True
            
        # Condition B: Lower bound rejection (Stop clustering)
        elif potential_ratio < reject_ratio:
            break
            
        # Condition C: The "Grey Zone" 
        # Check if the point is far enough from existing centers to warrant a new rule
        else:
            # Calculate distance to the closest existing center
            diff_to_centres = np.array(cluster_centres) - current_centre
            dist_to_centres_sq = np.sum(diff_to_centres ** 2, axis=1)
            min_dist = np.sqrt(np.min(dist_to_centres_sq))
            
            # Acceptance formula: Distance ratio + Potential ratio >= 1
            if (min_dist / radius) + potential_ratio >= 1.0:
                accept = True
            else:
                # Reject this specific point, set its potential to 0, and try the next highest
                potentials[max_idx] = 0.0
                highest_potential = np.max(potentials)
                max_idx = np.argmax(potentials)
                continue

        if accept:
            # Add the accepted point to our list of centers
            cluster_centres.append(current_centre)
            
            # 3. Subtractive Phase
            # Destroy the potential of all points near the new center so we don't pick them next
            diff = data - current_centre
            dist_sq = np.sum(diff ** 2, axis=1)
            subtraction_amount = highest_potential * np.exp(-beta * dist_sq)
            potentials = potentials - subtraction_amount
            
            # Prevent potential from going negative
            potentials = np.maximum(potentials, 0)
            
            # Find the next highest potential for the next loop iteration
            highest_potential = np.max(potentials)
            max_idx = np.argmax(potentials)

    return np.array(cluster_centres)


def clusters_to_rules(centres: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Converts mathematical cluster centers into human-readable fuzzy rules.

    Args:
        centres: The array of cluster centers.
        feature_names: List of column names corresponding to the features.

    Returns:
        A list of dictionaries representing explainable rules.
    """
    rules = []
    
    # Iterate through each center to formulate a logical premise
    for i, centre in enumerate(centres):
        premise = {}
        conditions = []
        
        # Map each feature's center value to its name
        for j, val in enumerate(centre):
            feature = feature_names[j]
            # Rounding to 4 decimal places for readability
            rounded_val = float(np.round(val, 4))
            premise[feature] = rounded_val
            conditions.append(f"{feature}~={rounded_val}")
            
        # Construct the explainable string
        description = f"IF {' AND '.join(conditions)} THEN suspicious"
        
        # Compile the final rule dict
        rule = {
            "rule_id": i + 1,
            "premise": premise,
            "description": description
        }
        rules.append(rule)
        
    return rules


def save_rules(rules: list[dict], filepath: str) -> None:
    """Saves the extracted fuzzy rules to a JSON file.

    Args:
        rules: The list of generated rule dictionaries.
        filepath: The destination path for the JSON file.
    """
    path = Path(filepath)
    
    # Create the parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the rules to disk with indentation for readability
    with open(path, 'w') as f:
        json.dump(rules, f, indent=4)