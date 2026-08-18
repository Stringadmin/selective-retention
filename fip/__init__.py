"""FIP: Feature Isolation for Plasticity in continual learning Transformers."""

from .feature_registry import FeatureRegistry, FeatureState, LayerRegistry
from .sparse_ffn import DenseFFN, FIPFFN
from .plasticity import PlasticityController
from .rollback import RollbackManager
from .optimizer import FeatureMaskedAdamW
from .non_ffn_protection import NonFFNProtector
from .lora import LoRADelta
from .model import FIPTransformer, ModelConfig

__all__ = [
    "FeatureRegistry",
    "FeatureState",
    "LayerRegistry",
    "DenseFFN",
    "FIPFFN",
    "PlasticityController",
    "RollbackManager",
    "FeatureMaskedAdamW",
    "NonFFNProtector",
    "LoRADelta",
    "FIPTransformer",
    "ModelConfig",
]
