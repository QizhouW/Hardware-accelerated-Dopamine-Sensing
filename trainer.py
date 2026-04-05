import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class SpectraTrainer:
    def __init__(self, loader, n_pca=8):
        self.loader = loader
        self.n_pca = n_pca

        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.X_train, self.Y_train = self.loader.get_train_data()
        self.X_test, self.Y_test = self.loader.get_test_data()
        all_X, all_Y, all_Y_reg = self.loader.get_all_data()

        # PCA
        self.pca = PCA(n_components=self.n_pca, svd_solver="full")
        self.pca.fit(all_X)
        self.pca_components = self.pca.components_

        # Min-max normalize PC
        self.pc_train = np.matmul(self.X_train, self.pca_components.T)

        self.pc_min = np.min(self.pc_train, axis=0)
        self.pc_max = np.max(self.pc_train, axis=0)
        self.pc_train_normalized = (self.pc_train - self.pc_min) / (
            self.pc_max - self.pc_min
        )

        # Transform test data
        self.pc_test = np.matmul(self.X_test, self.pca_components.T)
        self.pc_test_normalized = (self.pc_test - self.pc_min) / (
            self.pc_max - self.pc_min
        )

        # print(f'PCA explained variance: {self.pca.explained_variance_ratio_}')
        # print(f'Total variance explained: {np.sum(self.pca.explained_variance_ratio_):.4f}')

    def _build_model(self):
        self.model = Sequential(name="spectra_regression")
        self.model.add(Dense(32, activation="relu", input_shape=(self.n_pca,)))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    def train(self, epochs=100, batch_size=8, verbose=1):
        early_stopper = EarlyStopping(monitor="loss", patience=15)
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, verbose=0, patience=4, min_lr=1e-6
        )

        self.history = self.model.fit(
            self.pc_train_normalized,
            self.Y_train,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stopper],
        )

        return self.history

    def evaluate(self):
        Y_pred = self.model.predict(self.pc_test_normalized, verbose=0).flatten()

        # Inverse transform to original concentration scale
        Y_test_orig = np.array([self.loader.inverse_transform(y) for y in self.Y_test])
        Y_pred_orig = np.array([self.loader.inverse_transform(y) for y in Y_pred])

        # Calculate metrics: MSE on normalized Y, R2 on original scale
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(Y_test_orig, Y_pred_orig)

        print(f"\nTest MSE (normalized): {mse:.6f}")
        print(f"Test R2 (original scale): {r2:.6f}")
        return Y_pred, Y_test_orig, Y_pred_orig, mse, r2

    def evaluate_on_all_data(self):
        all_X, all_Y, all_Y_reg = self.loader.get_all_data()
        pc_all = np.matmul(all_X, self.pca_components.T)
        pc_all_normalized = (pc_all - self.pc_min) / (self.pc_max - self.pc_min)

        Y_pred = self.model.predict(pc_all_normalized, verbose=0).flatten()

        # Inverse transform to original concentration scale
        Y_orig = np.array([self.loader.inverse_transform(y) for y in all_Y_reg])
        Y_pred_orig = np.array([self.loader.inverse_transform(y) for y in Y_pred])

        # Calculate metrics: MSE on normalized Y, R2 on original scale
        mse = mean_squared_error(all_Y_reg, Y_pred)
        r2 = r2_score(Y_orig, Y_pred_orig)

        print(f"\nAll Data MSE (normalized): {mse:.6f}")
        print(f"All Data R2 (original scale): {r2:.6f}")

        return Y_pred, Y_orig, Y_pred_orig, mse, r2

    def visualize(self, mse, r2):
        Y_pred = self.model.predict(self.pc_test_normalized, verbose=0).flatten()
        Y_test_orig = np.array([self.loader.inverse_transform(y) for y in self.Y_test])
        Y_pred_orig = np.array([self.loader.inverse_transform(y) for y in Y_pred])

        # Get concentration list for x-axis
        c_list = self.loader.selected_concentrations
        x_ax = np.arange(np.min(c_list), np.max(c_list) + 1)

        plt.figure(figsize=(10, 8))
        plt.plot(x_ax, x_ax, "b", linewidth=2)
        plt.scatter(
            Y_test_orig, Y_pred_orig, marker="d", color="green", alpha=0.8, s=60
        )
        plt.legend(["Ideal fit", "Model Prediction"], fontsize=16)

        # Create tick labels
        ticks = []
        for c in x_ax:
            ticks.append("$10^{" + str(int(c)) + "}$")

        plt.xticks(x_ax, ticks, fontsize=12)
        plt.yticks(x_ax, ticks, fontsize=12)
        plt.title("mse={:.4f}, r2={:.4f}".format(mse, r2))

        return plt.gcf()

    def save_model(self, save_path):
        self.model.save(save_path)
        # np.save(save_path.replace('.h5', '_pca.npy'), self.pca_components)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from dataloader import SpectraDataLoader

    loader = SpectraDataLoader(
        data_dir="./data/dopamine_pbs",
        split_by_batch=False,
        concentration_list=[0, -11, -10, -9, -8, -7, -6],
        norm_type="anchor",
        test_ratio=0.25,
    )

    trainer = SpectraTrainer(loader=loader, n_pca=8)

    print("\nTraining model...")
    trainer.train(epochs=100, batch_size=8, verbose=1)

    print("\nEvaluating model...")
    Y_pred, Y_test_orig, Y_pred_orig, mse, r2 = trainer.evaluate()

    print("\nEvaluating on all data...")
    _ = trainer.evaluate_on_all_data()

    print("\nVisualizing results...")
    fig = trainer.visualize(mse, r2)
    plt.show()
