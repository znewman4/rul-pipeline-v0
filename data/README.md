# data/

| Folder | Contents | Version-controlled? |
|---|---|---|
| `bristolfe/` | forward-library exports from MATLAB/BristolFE (`.mat` v7 / v7.3, or `.npz`) | no (ignored by `bristolfe/.gitignore`) |
| `synthetic/` | cached toy libraries / synthetic measurements (`save_forward_library_npz`) | no (ignored by `synthetic/.gitignore`) |

Expected schema: see `src/rul_pipeline/io/README.md` and `matlab/export_forward_library.m`.
Load with `rul_pipeline.io.load_forward_library(path)`.

## confirmation_baselines/

Naming: `<experimental|simulated>_<quantity>[_<tool>].<ext>`, one folder per view.

| File | What it is |
|---|---|
| `experimental_fmc.mat` | real-specimen FMC (BRAIN `exp_data`, half matrix) |
| `experimental_tfm_brain.fig` | BRAIN's TFM figure of that real data |
| `experimental_tfm_brain.mat` | image/axes/colour map extracted from the `.fig` by `matlab/extract_fig_tfm.m` |
| `simulated_bristolfe_fmc.mat` | BristolFE-simulated FMC (made by `run_validation_fmc.m`; `above_view/` only) |
