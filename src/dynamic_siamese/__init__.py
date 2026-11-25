from .layers import ExternalKernelConv2d
from .generator import KernelGenerator
from .networks import DynamicSiameseNet, SiameseBaseline, ResNetTower
from .datasets import (SyntheticSimilarityDataset, TransformWrapper,
                       CUB200MetricLearning, Cars196MetricLearning, NABirdsMetricLearning,
                       AircraftMetricLearning,
                       MPerClassBatchSampler)
from .datasets_multi import (GenericMLDataset, CombinedBalancedDataset,
                             LOADERS as MULTI_LOADERS,
                             DATASET_URLS, DATASET_SUBDIRS)
from .utils import cosine_similarity, pair_accuracy
from .e2e_models import (
    build_e2e_model, E2E_ARCHS, load_baseline_encoder_into)
from .baselines import (BASELINE_ARCHS, DISPATCH as BASELINE_DISPATCH)
from .protonet import protonet_accuracy
from .hyper_hybrid import HyperHybridNet
from .hyper_multi import HyperMultiNet
from .hyper_comparator import (
    HyperComparisonNet, MultiKernelGenerator, ReferenceConditionedHead,
)

__all__ = [
    "ExternalKernelConv2d",
    "KernelGenerator",
    "DynamicSiameseNet",
    "SiameseBaseline",
    "ResNetTower",
    "SyntheticSimilarityDataset",
    "TransformWrapper",
    "CUB200MetricLearning",
    "Cars196MetricLearning",
    "NABirdsMetricLearning",
    "AircraftMetricLearning",
    "GenericMLDataset",
    "CombinedBalancedDataset",
    "MULTI_LOADERS",
    "DATASET_URLS",
    "DATASET_SUBDIRS",
    "MPerClassBatchSampler",
    "cosine_similarity",
    "pair_accuracy",
]

__version__ = "0.1.0"
