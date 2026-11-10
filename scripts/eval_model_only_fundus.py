#!/usr/bin/env python3
"""
Standalone runner script for eval_model_only_fundus.
Provides fovea-centered fundus cropping and 3D OCT volume synthesis without requiring OCT inputs.
"""
from pmrf3dsynthesis.eval_model_only_fundus import run_cli

if __name__ == "__main__":
    run_cli()
