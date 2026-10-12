"""
Deployment Step 01: Model & Artifact Loader
Verifies existence, loads preprocessor and model artifacts, and inspects signatures.
"""

import sys
from pathlib import Path
from typing import Tuple, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_utils import load_artifact, MODEL_PATH, PREPROCESSOR_PATH


def load_production_pipeline() -> Tuple[Any, Any]:
    """Loads and validates both preprocessor and trained credit scoring model."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor artifact at: {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model artifact at: {MODEL_PATH}")

    preprocessor = load_artifact(PREPROCESSOR_PATH)
    model = load_artifact(MODEL_PATH)
    return preprocessor, model


def run_load_verification():
    print("=" * 70)
    print(" CREDITFAIR DEPLOYMENT - STEP 01: ARTIFACT LOAD VERIFICATION")
    print("=" * 70)
    
    print(f"Loading preprocessor from: {PREPROCESSOR_PATH}...")
    print(f"Loading model from       : {MODEL_PATH}...")
    
    preprocessor, model = load_production_pipeline()
    
    print("\n[SUCCESS] Artifacts loaded successfully:")
    print(f"  * Preprocessor Type : {type(preprocessor).__name__}")
    print(f"  * Model Type        : {type(model).__name__}")
    print("=" * 70)
    return preprocessor, model


if __name__ == "__main__":
    run_load_verification()
