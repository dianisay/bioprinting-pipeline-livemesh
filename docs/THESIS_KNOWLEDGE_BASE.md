# Thesis Knowledge Base (Quick Retrieval)

## Document Identity
- **Author:** Diana Paola Ayala Roldan
- **Program:** Computational Sciences (Tecnologico de Monterrey)
- **Year:** 2025
- **Thesis title:** Convolutional Neural Network-Based Machine Learning for Three-Dimensional Motion Planning and Control in In-Situ Robotic Bioprinters for Superficial Tissue Regeneration
- **Main language:** English (with Spanish abstract)
- **Core claim:** A CNN-Transformer with a polar decoder enables autonomous image-to-trajectory wound bioprinting.

## One-Paragraph Executive Summary
The thesis proposes and validates an end-to-end computational pipeline for autonomous in-situ robotic bioprinting of superficial wounds. Starting from a single RGB wound image, a CNN-Transformer with a polar-parameterized decoder predicts an ordered closed-loop wound boundary, then a 3D reconstruction and conformal honeycomb infill module generate a normal-aligned deposition path. An 8-DOF robotic system (UR5 + XY gantry) executes the toolpath with closed-loop visual monitoring. Reported results show superior boundary performance for the polar decoder over Cartesian baselines, high wound coverage, and sub-millimeter RMS tracking error.

## Research Framing

### Problem Statement
- Current in-situ bioprinting remains semi-automated.
- Main gap is computational autonomy: perception + 3D planning + execution.
- Existing pipelines often depend on manual segmentation/calibration and do not guarantee valid closed-loop trajectories.

### Research Question
- Can a CNN-Transformer with a polar decoder generate closed-loop 3D deposition trajectories from visual input for autonomous wound treatment?

### Hypothesis
- Polar decoder will outperform parallel and autoregressive Cartesian decoders (better Chamfer/Hausdorff/IoU).
- Full integrated system will keep tracking error below 1 mm (simulation).

## Proposed System (6 Modules)
1. **Wound boundary detection**  
   CNN-Transformer (ResNet-50 + Transformer encoder + polar decoder) outputs ordered closed-loop boundary points.
2. **Depth estimation / 3D reconstruction**  
   Multi-view eye-in-hand reconstruction (preferred option in reported results).
3. **3D trajectory generation**  
   Conformal mapping + honeycomb lattice + TSP/MILP cell ordering + normal-aligned mapping back to 3D.
4. **Robot motion planning and control**  
   IK + manipulability optimization + collision checks + PID velocity control.
5. **Execution and real-time feedback**  
   Camera monitoring during printing and post-deposition verification.
6. **Validation**  
   In-silico and phantom-level evaluation with module and end-to-end metrics.

## Core Technical Innovation

### Polar output representation (main contribution)
- Predict centroid + radii at fixed angles.
- Guarantees:
  - ordered waypoints,
  - closed loop by construction,
  - graceful failure behavior.
- Known limitation: assumes roughly star-convex wounds.

## Data and Training (Module 1)
- Total training pairs: **2,934**
  - 934 from FUSeg (real),
  - 2,000 synthetic.
- Training setup highlights:
  - Adam, lr 1e-4,
  - batch size 8,
  - early stopping.
- Ablation compares three decoders with same encoder:
  - Parallel Cartesian (DETR-style),
  - Autoregressive Cartesian,
  - Polar (proposed).

## Key Quantitative Results

### Ablation (held-out test set)
- **Parallel Cartesian:** Chamfer 4.72 mm, Hausdorff 12.41 mm, IoU 0.71, closure 8.34 mm, ordering 23.1%.
- **Autoregressive Cartesian:** Chamfer 3.18 mm, Hausdorff 8.67 mm, IoU 0.79, closure 3.52 mm, ordering 81.4%.
- **Polar (proposed):** Chamfer 2.31 mm, Hausdorff 5.14 mm, IoU 0.91, closure 0.00 mm, ordering 100%.

### 3D reconstruction (reported)
- Mean surface RMS error: **0.38 mm**
- Max surface error: **1.12 mm**
- Mean depth MAE: **0.29 mm**
- Completeness: **94.7%**

### Trajectory generation (reported)
- Wound coverage: **97.2%**
- Travel-to-deposition ratio: **0.18**
- TSP/MILP travel reduction vs naive: **36.0%**

### Robot execution (reported)
- RMS tracking error: **0.41 mm**
- Mean orientation error: **0.8 deg**
- Hypothesis threshold (<1 mm RMS): **satisfied**

### End-to-end (in-silico)
- Full autonomous pipeline time: **4.2 min per wound (avg)**
- Post-deposition coverage: **95.8%**

### Phantom validation
- 5 phantoms tested.
- Mean boundary Chamfer: **3.14 mm**
- Coverage: **93.6%**
- RMS tracking error: **0.58 mm**

## PoC Baseline vs Proposed
- Baseline was segment-then-trace (U-Net + contour + G-code).
- Main improvements with proposed pipeline:
  - closure: 0.74 mm -> 0.00 mm,
  - wound coverage: 78% -> 97.2%,
  - autonomy: manual calibration -> autonomous pipeline.

## Answer to Thesis Claims
- **Research question:** Answered **Yes** (within simulation + phantom scope).
- **Hypothesis:** **Accepted** (all quantitative claims supported in reported results).

## Limitations (Explicitly Acknowledged)
- Star-convex assumption fails on highly concave/multi-lobed wounds.
- Simulation-to-reality gap remains significant.
- Single-cylinder parameterization not ideal for complex anatomy.
- Static wound assumption during execution.
- Bioink rheology not modeled in simulation.
- No biological validation (cell viability/tissue integration not evaluated).

## Future Work (Proposed in Thesis)
- Multi-lobe / non-star-convex boundary handling.
- Domain adaptation for clinical imaging.
- Multi-patch surface parameterization.
- Real-time geometry updates during deposition.
- Bioink-aware planning (rheology-informed control).
- Ex vivo -> animal -> clinical pilot progression.
- Dynamic collision avoidance in clinical environments.

## Notable Draft Markers To Revisit
- Several sections include placeholders and "NOTE TO FUTURE ME" reminders.
- Some figures/tables are marked as placeholders pending final images.
- Comparative literature claims should be re-checked for 2025-2028 concurrent work.
- Discussion contains at least one citation-needed marker for clinical procedure timing.
- Publications section is currently template text, not finalized entries.

## Fast Lookup Map (Ask Me Like This)
- **"What is the main contribution?"** -> Polar decoder + end-to-end modular autonomous pipeline.
- **"Which module does what?"** -> See "Proposed System (6 Modules)".
- **"What are the exact ablation numbers?"** -> See "Key Quantitative Results -> Ablation".
- **"Did the hypothesis pass?"** -> Yes; see "Answer to Thesis Claims".
- **"What are the biggest limitations?"** -> See "Limitations".
- **"How much better than baseline?"** -> See "PoC Baseline vs Proposed".
- **"What is validated physically?"** -> Phantom results (5 models), no full clinical validation.
- **"How long does end-to-end take?"** -> ~4.2 min/wound average (in-silico report).

## Technology Stack
- **Language:** 100% Python (no MATLAB). PoC was MATLAB; main thesis pipeline is unified Python.
- **Deep Learning:** PyTorch (CNN-Transformer, all three decoders)
- **Robotics:** roboticstoolbox-python (Peter Corke) or custom numpy for kinematics, scipy.optimize for IK
- **Optimization:** PuLP for MILP/TSP, scipy.milp as fallback
- **3D Reconstruction:** OpenCV + Open3D
- **Simulation:** CoppeliaSim via Python ZeroMQ API
- **Control:** PID implemented as pure Python class (numpy)
- **Visualization:** matplotlib + plotly

## Glossary (Quick)
- **CNN-Transformer:** Hybrid model combining convolutional feature extraction and self-attention context modeling.
- **Polar decoder:** Predicts radial boundary values at fixed angles around centroid.
- **Chamfer distance:** Average nearest-neighbor boundary error.
- **Hausdorff distance:** Worst-case boundary mismatch.
- **IoU:** Overlap metric between predicted and ground-truth wound area.
- **Conformal mapping:** Distortion-controlled mapping between curved surface and 2D parameter domain.
- **TSP/MILP (MTZ):** Optimization method for efficient cell visitation order.
- **Manipulability:** Jacobian-based dexterity measure; higher means safer distance from singularities.

## Practical Retrieval Notes
- When asked for any claim, return:
  1) the exact number,
  2) where it belongs in the pipeline/module,
  3) whether it is simulation or phantom evidence.
- If asked about clinical readiness, emphasize that current evidence is pre-clinical (simulation + phantom), not clinical trial validated.
