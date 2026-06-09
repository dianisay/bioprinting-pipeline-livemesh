"""
LiveMesh: Autonomous in-situ bioprinting pipeline.

Perception --> Reconstruction --> Toolpath --> Execution

Modules:
    livemesh.perception     CNN-Transformer encoder + polar/DETR/AR decoders
    livemesh.segmentation   U-Net wound segmentation
    livemesh.reconstruction Poisson / DeepCurrents surface reconstruction
    livemesh.toolpath       Geodesic, planar, honeycomb toolpath generation
    livemesh.robot          8-DOF robot model, IK solver, scaffold analysis
    livemesh.training       Training, evaluation, ablation pipelines
    livemesh.data           Datasets (FUSeg, synthetic 2D/3D, multi-view)
    livemesh.pipeline       End-to-end orchestrator
"""

__version__ = "0.2.0"
