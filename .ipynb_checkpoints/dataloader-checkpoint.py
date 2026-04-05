import numpy as np


class SpectraDataLoader:
    def __init__(
        self,
        data_dir,
        split_by_batch=True,
        concentration_list=None,
        norm_type="none",
        test_ratio=0.25,
        random_seed=None,
    ):
        """
        data_dir: path to data directory
        split_by_batch: True to split by batch (avoid leakage), False to split by item
        concentration_list: list of concentrations to use (e.g., [0, -11, -10, -9])
        norm_type: 'anchor', 'avg', or 'none' for spectra normalization
        test_ratio: ratio of test set
        random_seed: random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split_by_batch = split_by_batch
        self.concentration_list = concentration_list
        self.norm_type = norm_type
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self._load_data()
        self._filter_concentrations()
        self._normalize_spectra()
        self._prepare_regression_targets()
        self._split_data()

    def _load_data(self):
        if self.norm_type == "anchor":
            self.X_raw = np.load(f"{self.data_dir}/X_an.npy")
        elif self.norm_type == "avg":
            self.X_raw = np.load(f"{self.data_dir}/X_avg.npy")
        else:
            self.X_raw = np.load(f"{self.data_dir}/X.npy")

        self.Y_raw = np.load(f"{self.data_dir}/Y.npy")
        self.wls = np.loadtxt(f"{self.data_dir}/wavelengths.txt")

        self.n_concentrations, self.n_samples, self.n_wavelengths = self.X_raw.shape

    def _filter_concentrations(self):
        if self.concentration_list is None:
            self.concentration_indices = np.arange(self.n_concentrations)
            self.selected_concentrations = np.unique(self.Y_raw[:, 0])
        else:
            concentration_values = self.Y_raw[:, 0]
            self.concentration_indices = []
            for c in self.concentration_list:
                idx = np.where(concentration_values == c)[0]
                if len(idx) > 0:
                    self.concentration_indices.append(idx[0])
            self.concentration_indices = np.array(self.concentration_indices)
            self.selected_concentrations = np.array(self.concentration_list)

        self.X_filtered = self.X_raw[self.concentration_indices]
        self.Y_filtered = self.Y_raw[self.concentration_indices]

    def _normalize_spectra(self):
        # Nothing to to, can add some norm method later. But it is not that important
        self.X_normalized = self.X_filtered

    def _prepare_regression_targets(self):
        concentration_values = self.selected_concentrations.copy()

        # Handle 0 concentration: treat as min_concentration - 1
        non_zero_concs = concentration_values[concentration_values != 0]
        if len(non_zero_concs) > 0:
            min_conc = non_zero_concs.min()
            self.zero_replacement = min_conc - 1
        else:
            self.zero_replacement = -1

        # Replace 0 with zero_replacement for normalization
        transformed_concs = concentration_values.copy().astype(float)
        transformed_concs[concentration_values == 0] = self.zero_replacement

        # Normalize to [0, 1]
        self.y_min = transformed_concs.min()
        self.y_max = transformed_concs.max()

        # Build mapping
        self.concentration_to_normalized = {}
        for c in concentration_values:
            c_transformed = self.zero_replacement if c == 0 else c
            normalized_val = (c_transformed - self.y_min) / (self.y_max - self.y_min)
            self.concentration_to_normalized[c] = normalized_val

        # Apply transformation
        Y_flat = self.Y_filtered.flatten()
        Y_normalized = np.zeros_like(Y_flat, dtype=float)
        for i, c in enumerate(Y_flat):
            Y_normalized[i] = self.concentration_to_normalized[c]

        self.Y_normalized = Y_normalized.reshape(self.Y_filtered.shape)

    def _split_data(self):
        C, N = self.Y_normalized.shape

        if self.split_by_batch:
            # N = 30 = 10 batches * 3 measurements
            batch_size = 3
            n_batches = N // batch_size

            train_mask = np.ones((C, N), dtype=bool)

            for c_idx in range(C):
                batch_indices = np.arange(n_batches)
                n_test_batches = max(1, int(n_batches * self.test_ratio))
                test_batches = np.random.choice(batch_indices, n_test_batches, replace=False)

                for batch_idx in test_batches:
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    train_mask[c_idx, start_idx:end_idx] = False

            # Flatten and split
            X_flat = self.X_normalized.reshape(-1, self.n_wavelengths)
            Y_flat = self.Y_normalized.flatten()
            train_mask_flat = train_mask.flatten()

            self.X_train = X_flat[train_mask_flat]
            self.Y_train = Y_flat[train_mask_flat]
            self.X_test = X_flat[~train_mask_flat]
            self.Y_test = Y_flat[~train_mask_flat]

        else:  # item split
            # Split each concentration separately to ensure balanced split
            C, N = self.Y_normalized.shape
            train_mask = np.ones((C, N), dtype=bool)

            for c_idx in range(C):
                n_samples = N
                n_test = max(1, int(n_samples * self.test_ratio))
                test_indices = np.random.choice(n_samples, n_test, replace=False)
                train_mask[c_idx, test_indices] = False

            # Flatten and split
            X_flat = self.X_normalized.reshape(-1, self.n_wavelengths)
            Y_flat = self.Y_normalized.flatten()
            train_mask_flat = train_mask.flatten()

            self.X_train = X_flat[train_mask_flat]
            self.Y_train = Y_flat[train_mask_flat]
            self.X_test = X_flat[~train_mask_flat]
            self.Y_test = Y_flat[~train_mask_flat]

    def transform_to_regression(self, concentration_log):
        """Transform concentration (in log scale) to regression target [0,1]"""
        c_transformed = self.zero_replacement if np.isclose(concentration_log, 0) else concentration_log
        return (c_transformed - self.y_min) / (self.y_max - self.y_min)

    def inverse_transform(self, normalized_value):
        """Transform regression prediction [0,1] back to log concentration"""
        c_transformed = normalized_value * (self.y_max - self.y_min) + self.y_min
        if np.isclose(c_transformed, self.zero_replacement):
            return 0
        else:
            return c_transformed

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test

    def get_all_data(self):
        X = self.X_normalized.reshape(-1, self.n_wavelengths)
        Y = self.Y_filtered.flatten()
        Y_reg = self.Y_normalized.flatten()
        return X, Y, Y_reg


if __name__ == "__main__":
    # Usage example
    loader = SpectraDataLoader(
        data_dir="./data/dopamine_csf",
        split_by_batch=True,
        concentration_list=[0, -11, -10, -9, -8, -7, -6, -5, -4],
        norm_type="none",
        test_ratio=0.25,
        random_seed=42,
    )

    X_train, Y_train = loader.get_train_data()
    X_test, Y_test = loader.get_test_data()

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    print(f"Y_train range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
    print(f"Y_test range: [{Y_test.min():.4f}, {Y_test.max():.4f}]")

    # Test transformation functions
    test_conc = -9
    normalized = loader.transform_to_regression(test_conc)
    back = loader.inverse_transform(normalized)
    print(f"\nTransformation test: {test_conc} -> {normalized:.4f} -> {back:.4f}")

    import matplotlib.pyplot as plt

    for i in range(X_train.shape[0]):
        plt.plot(loader.wls[:], X_train[i], alpha=0.5, label="Sample Spectrum")
    plt.show()
