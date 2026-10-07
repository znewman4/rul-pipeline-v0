# data/

| Folder | Contents | Version-controlled? |
|---|---|---|
| `bristolfe/` | forward-library exports from MATLAB/BristolFE (`.mat` v7 / v7.3, or `.npz`) | no (ignored by `bristolfe/.gitignore`) |
| `synthetic/` | cached toy libraries / synthetic measurements (`save_forward_library_npz`) | no (ignored by `synthetic/.gitignore`) |

Expected schema: see `src/rul_pipeline/io/README.md` and `matlab/export_forward_library.m`.
Load with `rul_pipeline.io.load_forward_library(path)`.
