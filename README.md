# Robust Counterfactual Forecasting

Code and manuscript sources for the paper on robust counterfactual forecasting of post-policy air-quality deviations in Madrid.

## Repository contents

- `Code/`: data-preparation notebooks, download helper, experiment runners, and manuscript diagnostics.
- `Docs/NewTemplate/`: LaTeX source of the manuscript and the compiled PDF.

## Reproducibility

The raw Madrid datasets are not stored in this repository because they are too large for version control. They can be retrieved from the official Madrid Open Data Portal:

- https://datos.madrid.es/portal/site/egob/
- https://datos.madrid.es/dataset/201200-0-calidad-aire-horario
- https://datos.madrid.es/dataset/300352-0-meteorologicos-horarios
- https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico
- https://datos.madrid.es/dataset/202468-0-intensidad-trafico

The helper script `Code/download_madrid_open_data.py` downloads the official source files into the directory structure expected by the notebooks.

Main execution entry points:

- `Code/run_notebook_pipeline.py`
- `Code/run_counterfactual_experiments.py`
- `Code/run_robustness_extensions.py`
- `Code/run_manuscript_diagnostics.py`

## Manuscript

The current paper source is under `Docs/NewTemplate/`, and the latest compiled PDF is `Docs/NewTemplate/samplepaper.pdf`.
