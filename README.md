# Towards real-time additive-free dopamine detection at $10^{-8}$ mM with hardware accelerated platform integrated on camera

This repository contains the implementation, data processing pipelines, and optical accelerator models described in our work on ultra-sensitive dopamine (DA) detection. 

By integrating engineered **light-scattering membranes** and **optical accelerators** directly with commercial vision cameras, we achieve a detection limit of **$10^{-8}$ mM**—improving upon state-of-the-art electrochemical workstations by over two orders of magnitudes.

## 🌟 Key Features
* **Ultra-High Sensitivity:** Detection limits down to $10^{-8}$ mM for physiological neurotransmitters.
* **Additive-Free Platform:** No chemical additives required, ensuring minimal interference in complex biological matrices (e.g., uric and ascorbic acid).

### Prerequisites
* Python 3.8+
* Tensorflow 2.0+
* sklearn 1.5+


### Installation
```bash
git clone https://github.com/QizhouW/Hardware-accelerated-Dopamine-Sensing.git
```

### Train
Set up EXPERIMENT_CONFIGS in run.py, then

```bash
python run.py
```