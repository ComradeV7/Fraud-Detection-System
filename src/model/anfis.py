import torch
import torch.nn as nn
import numpy as np

class ANFIS(nn.Module):
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) model.
    It combines the explainable 'If-Then' logic of fuzzy systems with the gradient-based learning capabilities of neural networks.
    The architecture uses five specialized layers to compute membership, firing strength, normalization, and weighted consequences.
    """
    def __init__(self, n_inputs: int, n_rules: int):
        super().__init__()
        
        # Layer 1 parameters: Defines the fuzzy membership curves (Gaussian)
        self.centres = nn.Parameter(torch.empty(n_rules, n_inputs))
        self.widths = nn.Parameter(torch.ones(n_rules, n_inputs))
        
        # Layer 4 parameters: Defines the linear weights for the output consequence
        self.consequents = nn.Parameter(torch.empty(n_rules, n_inputs + 1))
        
        # Initialize empty parameters to standard normal distribution for stability
        nn.init.normal_(self.centres)
        nn.init.normal_(self.consequents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Step 1 (Layer 1): Gaussian membership — exp(-((x-centres)/widths)^2)
        # We unsqueeze x from (batch, inputs) to (batch, 1, inputs) so it broadcasts 
        # mathematically across the (rules, inputs) shape of our centres.
        x_expanded = x.unsqueeze(1)
        fuzzy_memberships = torch.exp(-((x_expanded - self.centres) / self.widths) ** 2)

        # Step 2 (Layer 2): Rule strength — product of memberships per rule
        # Represents the logical "AND" operation for the premise of each rule.
        # Shape becomes (batch, rules)
        rule_strengths = torch.prod(fuzzy_memberships, dim=2)

        # Step 3 (Layer 3): Normalise strengths — divide each by sum of all
        # We add a tiny epsilon (1e-8) to prevent division by zero errors during training.
        sum_strengths = torch.sum(rule_strengths, dim=1, keepdim=True)
        normalised_strengths = rule_strengths / (sum_strengths + 1e-8)

        # Step 4 (Layer 4): Weighted consequents — linear per rule using [x, 1]
        # We append a column of 1s to the input tensor to act as the bias term for the linear combination.
        ones = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)  # Shape: (batch, inputs + 1)
        
        # Matrix multiplication computes the linear consequence for every rule simultaneously.
        # x_aug @ consequents.T gives shape (batch, rules)
        rule_consequences = torch.matmul(x_aug, self.consequents.T)

        # Step 5 (Layer 5): Output sum — sum of normalised_strength * consequent, apply torch.sigmoid
        # This aggregates all rule decisions into a single continuous logit, then maps to a 0-1 probability.
        weighted_output = torch.sum(normalised_strengths * rule_consequences, dim=1)
        
        # Return shape: (batch_size,)
        return torch.sigmoid(weighted_output)

def initialise_centres(model: ANFIS, centres: np.ndarray) -> None:
    """
    Copies cluster centres numpy array into model.centres parameter.
    
    Args:
        model: The initialized ANFIS PyTorch model.
        centres: The numpy array of cluster centers extracted from subtractive clustering.
    """
    # Uses torch.no_grad() context to prevent PyTorch from tracking this injection in the computation graph
    with torch.no_grad():
        model.centres.copy_(torch.from_numpy(centres))