from captum.attr import Saliency
import numpy as np
import torch

class SaliencyAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
    
    def compute_saliency(self, input_tensor):
        """Compute saliency with respect to the reconstruction error"""
        self.model.eval()
        input_tensor = input_tensor.clone().detach().to(self.device).requires_grad_(True) # removes captum warning UserWarning: Input Tensor 0 did not already require gradients, required_grads has been set automatically.

        # Define a forward function that returns scalar error
        def forward_func(inputs):
            reconstructions = self.model(inputs)
            return torch.mean((inputs - reconstructions)**2, dim=1)
        
        saliency = Saliency(forward_func)
        return saliency.attribute(input_tensor).detach().cpu().numpy()
    
    def explain(self, input_tensor, top_k=3):
        """Generate numerical explanation"""
        saliency_values = self.compute_saliency(input_tensor)
        abs_saliency = np.abs(saliency_values)
        mean_importance = np.mean(abs_saliency, axis=0)
        
        top_indices = np.argsort(mean_importance)[::-1][:top_k]
        return {
            'top_features': [(self.feature_names[i], float(mean_importance[i]))
                           for i in top_indices],
            'saliency_values': saliency_values,
            'feature_importance': dict(zip(self.feature_names, mean_importance.tolist()))
        }