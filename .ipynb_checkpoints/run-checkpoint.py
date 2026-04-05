import os

import numpy as np
import pandas as pd
from dataloader import SpectraDataLoader
from tensorflow.keras.models import load_model
from trainer import SpectraTrainer


def train_and_save_all_models(
    material_type,
    data_dir,
    concentration_list,
    n_runs=100,
    epochs=200,
    n_pca=8,
    norm_type="anchor",
    base_dir="res",
):
    """
    Train multiple models and save all models

    Args:
        material_type: name of material (e.g., 'dopamine_csf')
        data_dir: path to data directory
        concentration_list: list of concentrations to use
        n_runs: number of training runs
        epochs: number of epochs per run
        n_pca: number of PCA components
        norm_type: normalization type
        base_dir: base directory for saving results
    """
    save_dir = f"{base_dir}/{material_type}"
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    results = []

    print(f"\n{'=' * 80}")
    print(f"Training {material_type}")
    print(f"Data: {data_dir}")
    print(f"Concentrations: {concentration_list}")
    print(f"Runs: {n_runs}, Epochs: {epochs}")
    print(f"{'=' * 80}\n")

    for run_idx in range(n_runs):
        print(f"[{run_idx + 1}/{n_runs}] Run {run_idx + 1}...", end=" ")

        # Create loader with different random seed each run
        loader = SpectraDataLoader(
            data_dir=data_dir,
            split_by_batch=False,
            concentration_list=concentration_list,
            norm_type=norm_type,
            test_ratio=0.25,
            random_seed=run_idx,
        )

        # Create trainer
        trainer = SpectraTrainer(loader=loader, n_pca=n_pca)

        # Train
        trainer.train(epochs=epochs, batch_size=8, verbose=0)

        # Evaluate
        Y_pred, Y_test_orig, Y_pred_orig, mse, r2 = trainer.evaluate()

        print(f"MSE={mse:.6f}, R2={r2:.6f}")

        # Save model immediately
        model_path = os.path.join(models_dir, f"run_{run_idx}.h5")
        trainer.save_model(model_path)

        # Store only metrics
        results.append({"run_idx": run_idx, "r2": r2, "mse": mse})

    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "results.csv")
    df_results.to_csv(csv_path, index=False)

    # Create sorted view for display
    df_sorted = df_results.sort_values("r2", ascending=False)

    print("\n" + "=" * 80)
    print(f"Results saved to {csv_path}")
    print("Top 3 models:")
    for i in range(min(3, len(df_sorted))):
        row = df_sorted.iloc[i]
        print(f"  {i + 1}. Run {row['run_idx']}: R2={row['r2']:.6f}, MSE={row['mse']:.6f}")
    print("=" * 80 + "\n")

    # Save summary
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Material: {material_type}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Concentrations: {concentration_list}\n")
        f.write(f"Total runs: {n_runs}\n")
        f.write(f"Epochs per run: {epochs}\n")
        f.write(f"N_PCA: {n_pca}\n")
        f.write("\nTop 3 models:\n")
        for i in range(min(3, len(df_sorted))):
            row = df_sorted.iloc[i]
            f.write(f"  {i + 1}. Run {row['run_idx']}: R2={row['r2']:.6f}, MSE={row['mse']:.6f}\n")

        f.write("\nAll runs statistics:\n")
        f.write(f"  Mean R2: {df_results['r2'].mean():.6f}\n")
        f.write(f"  Std R2: {df_results['r2'].std():.6f}\n")
        f.write(f"  Min R2: {df_results['r2'].min():.6f}\n")
        f.write(f"  Max R2: {df_results['r2'].max():.6f}\n")

    print(f"Summary saved to {summary_path}\n")

    return df_results


EXPERIMENT_CONFIGS = {
    "dopamine_csf": {
        "data_dir": "./data/dopamine_csf",
        "concentration_list": [-11, -10, -9, -8, -7, -6],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
    "dopamine_uric": {
        "data_dir": "./data/dopamine_uric",
        "concentration_list": [-11, -10, -9, -8, -7, -6],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
    "dopamine_pbs": {
        "data_dir": "./data/dopamine_pbs",
        "concentration_list": [-11, -10, -9, -8, -7, -6, -5, -4],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
}

EXPERIMENT_CONFIGS_W0 = {
    "dopamine_csf": {
        "data_dir": "./data/dopamine_csf",
        "concentration_list": [0, -11, -10, -9, -8, -7, -6],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
    "dopamine_uric": {
        "data_dir": "./data/dopamine_uric",
        "concentration_list": [0, -11, -10, -9, -8, -7, -6],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
    "dopamine_pbs": {
        "data_dir": "./data/dopamine_pbs",
        "concentration_list": [0, -11, -10, -9, -8, -7, -6, -5, -4],
        "split_by_batch": False,
        "norm_type": "anchor",
        "test_ratio": 0.2,
        "n_pca": 8,
    },
}


def load_and_evaluate(material_type, run_idx, base_dir="res", config_dict=None):
    """
    Load a trained model and evaluate it

    Args:
        material_type: name of material (e.g., 'dopamine_csf')
        run_idx: run index (0 to n_runs-1)
        base_dir: base directory where results are saved
        config_dict: config dictionary to use (if None, uses EXPERIMENT_CONFIGS)

    Returns:
        Dictionary with evaluation results
    """
    if config_dict is None:
        config_dict = EXPERIMENT_CONFIGS

    if material_type not in config_dict:
        raise ValueError(f"Unknown material_type: {material_type}")

    config = config_dict[material_type]

    # Recreate loader with same random seed
    loader = SpectraDataLoader(
        data_dir=config["data_dir"],
        split_by_batch=config["split_by_batch"],
        concentration_list=config["concentration_list"],
        norm_type=config["norm_type"],
        test_ratio=config["test_ratio"],
        random_seed=run_idx,
    )

    # Load model
    model_path = f"{base_dir}/{material_type}/models/run_{run_idx}.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path, compile=False)

    # Load PCA components
    pca_path = model_path.replace(".h5", "_pca.npy")
    pca_components = np.load(pca_path)

    # Get test data
    X_train, Y_train = loader.get_train_data()
    X_test, Y_test = loader.get_test_data()

    # Transform with PCA
    pc_train = np.matmul(X_train, pca_components.T)
    pc_min = np.min(pc_train, axis=0)
    pc_max = np.max(pc_train, axis=0)

    pc_test = np.matmul(X_test, pca_components.T)
    pc_test_normalized = (pc_test - pc_min) / (pc_max - pc_min)

    # Predict
    Y_pred = model.predict(pc_test_normalized, verbose=0).flatten()

    # Inverse transform
    Y_test_orig = np.array([loader.inverse_transform(y) for y in Y_test])
    Y_pred_orig = np.array([loader.inverse_transform(y) for y in Y_pred])

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test_orig, Y_pred_orig)

    print(f"\n{material_type} - Run {run_idx}:")
    print(f"  MSE (normalized): {mse:.6f}")
    print(f"  R2 (original scale): {r2:.6f}")

    return {
        "material_type": material_type,
        "run_idx": run_idx,
        "mse": mse,
        "r2": r2,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "Y_test_orig": Y_test_orig,
        "Y_pred_orig": Y_pred_orig,
        "loader": loader,
        "model": model,
    }


if __name__ == "__main__":
    n_runs = 1
    epochs = 200
    base_dir = "res"

    material_type = "dopamine_pbs"
    config = EXPERIMENT_CONFIGS[material_type]
    config["test_ratio"] = 0.2

    df_results = train_and_save_all_models(
        material_type=material_type,
        data_dir=config["data_dir"],
        concentration_list=config["concentration_list"],
        n_runs=n_runs,
        epochs=epochs,
        n_pca=config["n_pca"],
        norm_type=config["norm_type"],
        base_dir=base_dir,
    )
