# DVDPL

Official open-source implementation and benchmark visualization code for **DVDPL** (Virtual Decomposition Control-based Deep Physics-Informed Learning for Hydraulic Heavy-Duty Manipulator Dynamics).

> **Note on Open-Source Scope**: Please see [`Declaration.txt`](Declaration.txt). This repository releases the full code, trained checkpoints, ablation models, prior sensitivity experiments, and evaluation/plotting scripts for the proposed **DVDPL** method. Code for baseline methods from collaborating laboratories/colleagues (FFN, DeLaN), parameter identification (SID), and external force estimation is not publicly released.

---

## Repository Structure

```text
.
├── Declaration.txt
├── LICENSE
├── .gitattributes                  # Git LFS tracking for *.mat datasets
├── README.md
├── DVDPL/                          # Python (PyTorch) training & evaluation code
│   ├── Data/                       # Processed training/testing .mat datasets (Git LFS)
│   │   ├── Data_Train_end_250s.mat
│   │   ├── Data_Train_750s.mat
│   │   ├── Data_Train_DeLAN_end_250s.mat
│   │   ├── Data_Test_185s.mat
│   │   └── Data_Test_draw8_96s.mat
│   ├── models/                     # Pre-trained checkpoints, histories & .mat parameters (5 seeds)
│   ├── prior_sensitivity_results/  # CAD prior sensitivity checkpoints & JSON summary
│   ├── dynamics.py                 # Differentiable VDC rigid-body & LuGre friction solver
│   ├── net_layer.py
│   ├── activation_functions.py
│   ├── network_fcscale_inertia_cad.py             # Proposed DVDPL network
│   ├── network_cad_fixed.py                       # Ablation 1: DVDPL-CAD (fixed CAD inertial params)
│   ├── network_no_cad_prior.py                    # Ablation 2: DVDPL-NP (no CAD prior bounds)
│   ├── network_fcscale_inertia_cad_sensitivity.py # Prior width sensitivity network (k_m, k_r)
│   ├── train_chunk.py              # Train proposed DVDPL (250s / 750s, 5 seeds)
│   ├── train_cad_fixed.py          # Train ablation DVDPL-CAD (250s / 750s, 5 seeds)
│   ├── train_no_cad_prior.py       # Train ablation DVDPL-NP (250s / 750s, 5 seeds)
│   ├── train_prior_sensitivity.py  # Run CAD prior sensitivity experiment (k_m=k_r in {0.25, 0.50, 0.75})
│   ├── predict.py                  # Generate Test1/Test2/Test3 predictions for DVDPL
│   ├── predict_fixed.py            # Generate predictions for DVDPL-CAD
│   └── predict_no_cad_prior.py     # Generate predictions for DVDPL-NP
└── Plot/                           # MATLAB evaluation & figure generation scripts + prediction outputs
    ├── Print_All_Results.m         # Print Table II, Table III, Table IV (Mean ± SD over 5 seeds)
    ├── Plot_Test1_185s_single_seed.mlx
    ├── Plot_Test2_185s_single_seed.mlx
    ├── Plot_Test3_96s_single_seed.mlx
    ├── Plot_DataPre1.mlx
    ├── Plot_DataPre2.mlx
    ├── Plot_Test_Pd.mlx
    └── *.xlsx / *.mat              # Saved predictions and trajectory files
```

---

## Requirements & Installation

### 1. Git LFS (Required for `.mat` Datasets)
Several `.mat` files in `DVDPL/Data/` exceed 100 MB and are tracked via **Git LFS**. Make sure Git LFS is installed before cloning or pulling:

```bash
git lfs install
git clone https://github.com/wwdk1998/DVDPL.git
cd DVDPL
git lfs pull
```

### 2. Python Environment (for `DVDPL/`)
- Python 3.8+ (CUDA-capable GPU recommended)
- Core dependencies:
  - `torch`
  - `numpy`
  - `scipy`
  - `h5py`
  - `pandas`
  - `openpyxl`
  - `matplotlib`

### 3. MATLAB Environment (for `Plot/`)
- MATLAB (tested on R2023b; Curve Fitting Toolbox is used in `Plot_DataPre2.mlx` for `csaps`/`fnval`).

---

## Usage

### Step 1: Training (Optional — Pre-trained Models Already Provided in `DVDPL/models/`)
Navigate to `DVDPL/` (or run from repository root; all scripts resolve paths relative to their own location):

```bash
cd DVDPL
python train_chunk.py              # Proposed DVDPL (set TRAIN_MODEL_TYPE = '250s' or '750s' inside script)
python train_cad_fixed.py          # Ablation 1: DVDPL-CAD
python train_no_cad_prior.py       # Ablation 2: DVDPL-NP
python train_prior_sensitivity.py  # CAD prior width sensitivity analysis (k_m, k_r)
```

### Step 2: Prediction & Exporting Excel Results to `Plot/`
```bash
cd DVDPL
python predict.py                  # Outputs VDCDL_Test*.xlsx to ../Plot/
python predict_fixed.py            # Outputs VDCDL_Test*_fixed_*.xlsx to ../Plot/
python predict_no_cad_prior.py     # Outputs NoCADPrior_Test*.xlsx to ../Plot/
```

### Step 3: Benchmark Tables & Plotting in MATLAB (`Plot/`)
1. Open MATLAB and **set the Current Working Directory (Current Folder) to `Plot/`** before running any script.
   - **Important for `.mlx` Live Scripts**: When opening `Plot/*.mlx` in the MATLAB Live Editor and clicking **Run**, if your current working directory is not `Plot/`, please accept MATLAB's **"Change Folder"** prompt so relative paths (`'.'` and `fullfile('..', 'DVDPL', 'Data')`) resolve properly.
2. Run [Print_All_Results.m](Plot/Print_All_Results.m) in MATLAB to reproduce the quantitative tables (Mean ± SD across 5 random seeds):
   ```matlab
   cd('Plot')
   Print_All_Results
   ```
3. Run `Plot_Test1_185s_single_seed.mlx`, `Plot_Test2_185s_single_seed.mlx`, `Plot_Test3_96s_single_seed.mlx`, `Plot_DataPre1.mlx`, `Plot_DataPre2.mlx`, or `Plot_Test_Pd.mlx` in MATLAB to reproduce the manuscript figures.
