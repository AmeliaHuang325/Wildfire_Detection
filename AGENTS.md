# Repository Guidelines

This repository revolves around a YOLO-based wildfire detection workflow maintained through four core notebooks and curated datasets. Follow the guidance below to keep contributions consistent and reproducible.

## Project Structure & Module Organization
- `00_Image labeling.ipynb` to `03_Test.ipynb` are sequential; edit them in numerical order to avoid path drift.
- Training data lives under `datasets/Whildfire photos` with `images/` and `labels/` split into `train`, `val`, and `test`; configuration stays in `datasets/Whildfire photos/wildfire.yaml`.
- Raw Kaggle assets live in `Wildfire photos/`; keep newly downloaded material there before conversion.
- Model artifacts and experiment outputs are written to `runs/`; archive older runs into `archive/` when the folder gets large.

## Build, Test, and Development Commands
- `python -m venv .venv; .\.venv\Scripts\Activate.ps1` to isolate dependencies.
- `pip install -r requirements.txt` or `pip install ultralytics opencv-python numpy pandas torch torchvision` to match the notebooks.
- `jupyter notebook 02_Model training.ipynb` launches the primary training workflow; verify `wildfire.yaml` paths before running cells.
- `python -m ultralytics detect predict model=yolo11n.pt source=forest_fire.jpg` is a quick CLI smoke test mirroring the testing notebook.

## Coding Style & Naming Conventions
- Use Python 3.10+ syntax and PEP 8 spacing inside notebook cells; prefer explicit functions for data transforms so they can be reused.
- Name new notebooks or scripts with a zero-padded prefix (`04_`) to preserve chronological ordering.
- Keep file and directory names lowercase with underscores; YOLO label files must mirror their image basename.
- Run `black -l 100` and `ruff check` on any extracted `.py` utilities before committing.

## Testing Guidelines
- Notebook-based validation happens in `03_Test.ipynb`; ensure both qualitative plots and reported precision/recall are updated.
- For dataset updates, re-run `02_Model training.ipynb` end-to-end and attach loss/metric screenshots from `runs/detect/train`.
- Store verification images (before/after detections) in `runs/` rather than the dataset folders to avoid data leakage.

## Commit & Pull Request Guidelines
- Follow the existing imperative tense history (`Update README.md`, `reorganize the document`). Summaries should stay under 50 characters with detail in the body.
- Each PR should describe the dataset snapshot, notebook cells executed, and how outputs were validated; link the relevant issue or experiment ID.
- Include before/after detection imagery or metrics tables when accuracy is affected. Mention any new weights added to `runs/` and whether large files need Git LFS.

## Data & Environment Notes
- Keep raw Kaggle archives zipped in `Wildfire photos/Annotations` to preserve provenance; never commit proprietary data.
- When swapping YOLO weight files (e.g., `yolov8n.pt`, `yolo11n.pt`), record the source and license in the PR description.
