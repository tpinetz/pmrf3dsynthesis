# pmrf3dsynthesis

pmrf3dsynthesis provides tools to perform Fundus-to-MIP registration and prepare inputs for a 3D synthesis pipeline.

## Features
- Fundus to MIP registration (CLI) via [`pmrf3dsynthesis.cli_fundus_oct`](pmrf3dsynthesis/cli_fundus_oct.py)
- Registration logic in [`pmrf3dsynthesis.registration.registration`](pmrf3dsynthesis/registration/registration.py)
- Utility scripts such as [scripts/generate_pngs_for_rflow.py](scripts/generate_pngs_for_rflow.py)
- Project metadata and entry points: [pyproject.toml](pyproject.toml), [pmrf3dsynthesis.egg-info/entry_points.txt](pmrf3dsynthesis.egg-info/entry_points.txt)

## Installation
Install from source:
```sh
pip install -e .
```

## Usage

```sh
run-registration --help
```
This will display the help message with all available options for the `run-registration` command.

To run model inference, use:
```sh
eval-model --help
```

```sh
run-registration --help
```
This will display the help message with all available options for the `run-registration` command.

``` 
eval-model --help
```

runs model inference for a specific model.

## Project layout
- `pmrf3dsynthesis/` — package entrypoints and modules, e.g. `pmrf3dsynthesis/__init__.py`
- `pmrf3dsynthesis/registration/` — registration implementation (`registration.py`)
- `pmrf3dsynthesis/models/` — model code
- `scripts/` — helper scripts
- `notebooks/` — demo and evaluation notebooks

## Contributing
PRs welcome. Please follow existing code style, include unit tests for new functionality, and add descriptive commit messages.

## License
This project is licensed under the MIT License — see `LICENSE`.
