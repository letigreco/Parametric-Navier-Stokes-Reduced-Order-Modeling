# Parametric Navier–Stokes Reduced-Order Modeling

Comparison of **POD-Galerkin**, **POD Neural Networks (PODNN)** and **Physics-Informed Neural Networks (PINNs)** for a stationary parametric Navier–Stokes problem.

The project was developed for the **Model Order Reduction and Machine Learning** course at Politecnico di Torino.

## Project Overview

The considered problem is a stationary Navier–Stokes system defined on a two-dimensional domain and parameterized by:

- the kinematic viscosity;
- the frequency and amplitude of the external forcing.

A Full Order Model is constructed using Taylor–Hood finite elements. Three reduced or surrogate approaches are then implemented and compared:

1. **POD-Galerkin**
2. **POD Neural Network**
3. **Physics-Informed Neural Network**

The comparison focuses on approximation accuracy, computational cost and online speedup.

## Methods

### Full Order Model

The Full Order Model uses:

- Taylor–Hood P2–P1 finite elements;
- Newton iterations for the nonlinear convective term;
- a manufactured solution for convergence verification;
- a finite-element system with 6,761 degrees of freedom.

### POD-Galerkin

The POD-Galerkin reduced model is constructed from 450 parameter-dependent snapshots.

The implementation includes:

- separate POD bases for velocity and pressure;
- supremizer enrichment for reduced inf-sup stability;
- Galerkin projection of the full-order operators;
- a reduced Newton solver for the nonlinear system.

### POD Neural Network

The PODNN approach learns the map between the physical parameters and the POD coefficients.

The neural network:

- receives the parameter vector as input;
- predicts the reduced coefficients;
- reconstructs the solution using the POD basis;
- avoids solving the reduced nonlinear system during the online phase.

### Physics-Informed Neural Network

The PINN directly approximates velocity and pressure using the Navier–Stokes equations as training constraints.

The loss function combines:

- momentum-equation residuals;
- incompressibility residuals;
- boundary conditions;
- pressure normalization.

Different weights for the PDE residual were tested and compared.

## Key Results

The three approaches show different trade-offs between accuracy and computational efficiency:

| Method | Mean relative error | Approximate speedup |
|---|---:|---:|
| POD-Galerkin | 3.87 × 10⁻⁴ | 1.4× |
| PODNN | 2.61 × 10⁻² | 17.5× |
| PINN, λ = 0.1 | 7.33 × 10⁻¹ velocity error | 467× |

POD-Galerkin provides the highest accuracy, while PODNN offers a favorable compromise between accuracy and online computational cost. The PINN has the fastest evaluation but requires a substantially longer training phase and produces higher approximation errors.

## Repository Structure

```text
parametric-navier-stokes-rom/
├── README.md
│   └── Project overview, installation instructions and repository documentation.
│
├── MORandML_ProjectReport_GRECO_s336195.pdf
│   └── Complete report containing the mathematical formulation, methodology,
│       numerical experiments and comparison of the implemented approaches.
│
├── main_notebook.ipynb
│   └── Main workflow for the Full Order Model, snapshot generation,
│       POD-Galerkin, PODNN and PINN analyses.
│
├── other_utilities.py
│   └── Auxiliary functions for finite-element computations,
│       post-processing and visualization.
│
├── images/
│   └── Selected figures from the numerical experiments, including convergence,
│       POD eigenvalue decay, training losses and solution comparisons.
│
├── podnn/
│   └── Scripts and final results related to the POD Neural Network
│       architecture and hyperparameter experiments.
│
└── pinn_lambdas/
    └── Trained PINN models and loss histories obtained using different
        weights for the physics-informed loss.
```

Large intermediate datasets, generated snapshots, reference solutions, VTK exports, temporary files and runtime logs are not included in the repository.

## Requirements

The project was developed and tested under **Ubuntu through WSL**.

Main dependencies:

- Python
- NumPy
- SciPy
- Matplotlib
- PyTorch
- PyPolyDiM
- Jupyter Notebook

The standard Python dependencies can be installed with:

```bash
pip install numpy scipy matplotlib torch jupyter
```

PyPolyDiM must be installed separately according to its installation instructions and system requirements.

## Running the Project

Open the project folder inside the WSL/Ubuntu environment and start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
main_notebook.ipynb
```

and run the cells in order.

The complete execution includes:

- Full Order Model assembly and solution;
- snapshot generation;
- POD basis construction;
- POD-Galerkin evaluation;
- PODNN training and evaluation;
- PINN training and comparison.

Some stages, particularly snapshot generation and neural-network training, may require significant computational time.

## Generated Data

Large generated files are not included in the repository, including:

- Full Order Model snapshots;
- PODNN training datasets;
- reference solutions;
- VTK exports;
- temporary Python files;
- runtime logs.

These files can be regenerated by executing the relevant sections of the notebook.

Recommended exclusions include:

```text
__pycache__/
Export/
data/
podnn_data.npz
fom_reference.npz
fom_reference_mesh.npz
test_mu_list.npy
```

## Report

The complete mathematical formulation, implementation details, numerical experiments and discussion are available in:

```text
MORandML_ProjectReport_GRECO_s336195.pdf
```

## Author

Letizia Greco  
Politecnico di Torino  
Model Order Reduction and Machine Learning  
Academic Year 2025/2026
