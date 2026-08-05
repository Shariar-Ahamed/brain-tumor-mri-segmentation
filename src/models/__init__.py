from .baselines import build_baseline_model
from .proposed_hybrid import ProposedUnifiedHybridModel

def get_model(model_name="Proposed-Hybrid", in_channels=4, num_classes=4):
    if model_name == "Proposed-Hybrid":
        return ProposedUnifiedHybridModel(in_channels=in_channels, num_classes=num_classes)
    else:
        return build_baseline_model(model_name=model_name, in_channels=in_channels, classes=num_classes)

__all__ = [
    'get_model',
    'build_baseline_model',
    'ProposedUnifiedHybridModel'
]
