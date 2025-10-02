#!/usr/bin/env python3
"""
Carbon Footprint Measurement Script for Recommender Systems.

This script runs model experiments from the models/ directory and tracks their
carbon footprint using CodeCarbon.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from codecarbon import EmissionsTracker


def get_available_models(models_dir="models"):
    """
    Get a list of available model scripts in the models directory.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        List of available model scripts (without .py extension)
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Error: Models directory '{models_dir}' does not exist.")
        return []
    
    # Find all Python files in the models directory
    model_files = list(models_path.glob("*.py"))
    
    # Filter out __init__.py and return model names without extension
    models = [f.stem for f in model_files if f.name != "__init__.py"]
    
    return sorted(models)


def run_model_with_tracking(model_name, models_dir="models"):
    """
    Run a model script and track its carbon emissions.
    
    Args:
        model_name: Name of the model to run (without .py extension)
        models_dir: Path to the models directory
        
    Returns:
        Dictionary with emission measurements
    """
    model_path = Path(models_dir) / f"{model_name}.py"
    
    if not model_path.exists():
        print(f"Error: Model script '{model_path}' does not exist.")
        sys.exit(1)
    
    print("=" * 70)
    print(f"Running model: {model_name}")
    print(f"Script path: {model_path}")
    print("=" * 70)
    print()
    
    # Initialize the emissions tracker
    tracker = EmissionsTracker(
        project_name=f"recommender_system_{model_name}",
        output_dir=".",
        output_file="emissions.csv",
        log_level="warning"
    )
    
    # Start tracking emissions
    tracker.start()
    
    try:
        # Run the model script using subprocess
        result = subprocess.run(
            [sys.executable, str(model_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output from the model script
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"\nWarning: Model script exited with code {result.returncode}")
        
    except Exception as e:
        print(f"Error running model: {e}")
        tracker.stop()
        sys.exit(1)
    
    # Stop tracking and get emissions data
    emissions = tracker.stop()
    
    return {
        "model_name": model_name,
        "emissions_kg": emissions,
        "returncode": result.returncode
    }


def print_emissions_report(measurements):
    """
    Print a formatted report of the emission measurements.
    
    Args:
        measurements: Dictionary containing emission measurements
    """
    print()
    print("=" * 70)
    print("CARBON FOOTPRINT MEASUREMENT REPORT")
    print("=" * 70)
    print(f"Model: {measurements['model_name']}")
    print(f"CO2 Emissions: {measurements['emissions_kg']:.6f} kg")
    print(f"CO2 Emissions: {measurements['emissions_kg'] * 1000:.6f} g")
    print(f"Exit Code: {measurements['returncode']}")
    print("=" * 70)
    
    # Print additional context
    print("\nNote: Emissions are calculated based on:")
    print("  - Energy consumption during model execution")
    print("  - Carbon intensity of the electricity grid")
    print("  - Hardware specifications and utilization")
    print()


def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="Run recommender system models and track their carbon footprint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model svd_model
  %(prog)s --list
  %(prog)s --model svd_model --models-dir custom_models/
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model to run (without .py extension)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model scripts (default: models)"
    )
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        available_models = get_available_models(args.models_dir)
        print("Available models:")
        if available_models:
            for model in available_models:
                print(f"  - {model}")
        else:
            print("  No models found.")
        sys.exit(0)
    
    # Check if model is specified
    if not args.model:
        parser.print_help()
        print("\nError: Please specify a model to run with --model or use --list to see available models.")
        sys.exit(1)
    
    # Run the model with carbon tracking
    measurements = run_model_with_tracking(args.model, args.models_dir)
    
    # Print the emissions report
    print_emissions_report(measurements)
    
    # Exit with the same code as the model script
    sys.exit(measurements['returncode'])


if __name__ == "__main__":
    main()
