# JAX Inventory Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)

> **Enterprise-grade inventory optimization platform powered by JAX for accelerated computation and intelligent decision-making.**

---

## Core Features

### Scalable ML Infrastructure
- **JAX-Accelerated Computation** — End-to-end GPU/TPU acceleration with JIT compilation and automatic differentiation.  
- **Distributed Training Framework** — Supports data, model, and hybrid parallelism for large-scale optimization workloads.  
- **Vectorized Workflows** — Batch-parallel computation across SKU portfolios for efficient resource utilization.

### Production-Ready Engineering Stack
- **Modular API Services** — FastAPI-based microservices for inference, simulation, and scheduling.  
- **Monitoring & Reliability** — Integrated Prometheus metrics, structured logging, and latency profiling.  
- **Automated Experimentation** — MLflow / W&B tracking pipelines for reproducible research and deployment.

---

## Optimization Framework

### Hybrid Paradigm Design
The system combines **classical inventory control** with **machine learning** and **reinforcement learning**:
- **Traditional Methods:** EOQ, Safety Stock, and (s, S) models for stable environments.  
- **ML Layer:** LSTM and Transformer-based demand forecasting for dynamic patterns.  
- **RL Layer:** DQN agent for adaptive policy control in stochastic environments.

---

## Quick Start

### Installation
```bash
git clone https://github.com/kevinlmf/JAX_Inventory_Optimizer
cd JAX_Inventory_Optimizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Setup
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

### Run Demo
```bash
# Run all demos
./run_all_demos.sh

# Start API server
uvicorn src.api.main:app --reload
```

---

## Examples

```bash
# Compare all optimization methods (Traditional + ML + RL)
python experiments/compare_all_methods.py

# Enterprise demo: Cost optimization + Risk management
python experiments/demo_enterprise.py

# API service demonstration
python experiments/demo_api.py
```

More examples are available in [`examples/`](examples/README.md).

---

## Architecture

```
JAX_Inventory_Optimizer/
├── src/
│   ├── core/                    # Core framework
│   ├── methods/                 # Optimization algorithms
│   │   ├── traditional/         # EOQ, Safety Stock
│   │   ├── ml_methods/          # LSTM, Transformer
│   │   └── rl_methods/          # DQN agent
│   ├── cost_optimization/       # Financial analytics
│   ├── risk_management/         # Anomaly detection
│   ├── distributed/             # Multi-GPU training
│   ├── api/                     # FastAPI service
│   └── data/                    # Data management
├── experiments/                 # Demo scripts
├── k8s/                         # Kubernetes manifests
├── helm/                        # Helm charts
└── requirements.txt
```

---

## Performance Summary

| Metric | Value | Description |
|--------|-------|-------------|
| Computation Speed | 10–100× faster | JAX vs NumPy (GPU) |
| JIT Optimization | 50–100× faster | Portfolio optimization |
| Inference Latency | < 10 ms | Per recommendation |
| Distributed Scaling | 7.2× on 8 GPUs | Data parallelism |
| Cost Reduction | 20–35% | Real-world benchmarks |

---

## Method Comparison

| Method | Training | Inference | Adaptability | Best Use Case |
|--------|-----------|------------|---------------|----------------|
| EOQ | None | < 1 ms | Low | Stable demand |
| Safety Stock | None | < 1 ms | Medium | Service target control |
| LSTM | 1–5 min | < 10 ms | High | Complex temporal patterns |
| DQN | 10–60 min | < 5 ms | Very High | Dynamic stochastic systems |

---

## From Research to Production: End-to-End ML Lifecycle

The JAX Inventory Optimizer implements a complete machine learning lifecycle — from data ingestion to deployment and continuous self-improvement. The framework integrates distributed JAX training, retraining pipelines, and real-time monitoring to enable a fully autonomous optimization engine.

| Stage | Objective | Tools / Frameworks | Future Direction |
|:------:|------------|--------------------|------------------|
| **1. Data Preparation** | Collect and preprocess structured demand and inventory data | `pandas`, `Airflow`, `SQL` | Automated ingestion pipelines and centralized feature store |
| **2. Model Training** | Build and validate forecasting or control models | `PyTorch`, `JAX`, `scikit-learn` | Distributed JAX training with asynchronous updates |
| **3. Evaluation & Optimization** | Compare experiments and select optimal models | `Weights & Biases`, `MLflow` | Automated hyperparameter tuning and advanced experiment tracking |
| **4. Deployment** | Package and serve scalable inference APIs | `Docker`, `FastAPI` | Load-balanced endpoints with caching and latency monitoring |
| **5. Monitoring & Maintenance** | Automate retraining and detect model drift | `Kubernetes`, `Prometheus`, `CI/CD` | Real-time drift detection and continuous retraining pipelines |

Together, these stages form a closed-loop system — from development to production and back — enabling the optimizer to evolve continuously under real-world dynamics.

---

<div align="center"> May all our lives keep optimizing — like sleeping better lol. 💤 </div>




