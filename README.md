# JAX Inventory Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)

> **Enterprise-grade inventory optimization platform powered by JAX for accelerated computation and intelligent decision-making.**

## Overview

JAX Inventory Optimizer addresses the fundamental challenge in supply chain management: minimizing inventory costs while maintaining service levels through hardware-accelerated computation and intelligent algorithms.

---
## Core Features

## Engine

### Scalable ML Infrastructure
- **JAX-Accelerated Computation** — End-to-end GPU/TPU acceleration with JIT compilation and automatic differentiation.  
- **Distributed Training Framework** — Supports data, model, and hybrid parallelism (7.2× scaling on 8 GPUs).  
- **Vectorized Workflows** — Batch-parallel optimization across large-scale SKU portfolios for efficient resource utilization.

### Production-Ready Engineering Stack
- **Modular API Services** — FastAPI-based microservices for inference, simulation, and scheduling.  
- **Monitoring & Reliability** — Integrated Prometheus metrics, logging, and latency profiling for robust deployment.  
- **Automated Experimentation** — Full MLflow / W&B tracking pipeline with reproducible experiment orchestration.

---

## Methods

### Integrated Optimization Framework
- **Hybrid Paradigm Design** — Combines classical inventory theory with machine learning and control-based optimization.  
- **Machine Learning Layer** — LSTM and Transformer architectures for demand forecasting and temporal representation learning.  
- **Reinforcement Learning Layer** — DQN-based adaptive policy control for real-time decision optimization.

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

## Usage Examples

### Run Demo Scripts

```bash
# Compare all optimization methods (Traditional + ML + RL)
python experiments/compare_all_methods.py
# Output: results/comparisons/method_comparison_*.csv + performance plots

# Enterprise features: Cost optimization + Risk management
python experiments/demo_enterprise.py
# Features: Deadstock detection, cash flow prediction, anomaly detection

# API service demonstration
python experiments/demo_api.py
# Test: Model recommendations, batch processing, health checks

# MLOps: Experiment tracking + GPU profiling
python experiments/demo_mlops.py
# Integration: Weights & Biases tracking, performance profiling

# Distributed training (requires multiple GPUs)
python experiments/distributed_training.py --strategy auto --profile
# Strategies: data/model/hybrid parallelism
```

### Hands-on Examples

```bash
# 01: Basic usage - Traditional methods (EOQ, Safety Stock, s-S)
python examples/01_basic_usage.py
# Learn: Core concepts, inventory states, ordering decisions

# 02: ML forecasting - LSTM demand prediction
python examples/02_ml_forecasting.py
# Learn: Neural network training, complex pattern recognition

# 03: Cost optimization - Enterprise features
python examples/03_cost_optimization.py
# Features: JIT optimization (50-100x speedup), deadstock detection, cash flow

# 04: API client - Production integration
python examples/04_api_client.py
# Requires: uvicorn src.api.main:app --reload (in another terminal)
```

See [examples/README.md](examples/README.md) for detailed documentation.

---

## Architecture

```
JAX_Inventory_Optimizer/
├── src/
│   ├── core/                    # Core interfaces and framework
│   ├── methods/                 # Optimization algorithms
│   │   ├── traditional/         # EOQ, Safety Stock, (s,S)
│   │   ├── ml_methods/          # LSTM, Transformer
│   │   └── rl_methods/          # DQN agent
│   ├── cost_optimization/       # Financial analytics
│   ├── risk_management/         # Anomaly detection
│   ├── distributed/             # Multi-GPU training
│   ├── api/                     # FastAPI service
│   └── data/                    # Data management
├── experiments/                 # Demo scripts
├── k8s/                        # Kubernetes manifests
├── helm/                       # Helm charts
└── requirements.txt
```

---

## Performance Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Computation Speed | 10-100x faster | JAX vs NumPy (GPU) |
| JIT Optimization | 50-100x faster | Portfolio optimization |
| Inference Latency | < 10ms | Per recommendation |
| Distributed Scaling | 7.2x on 8 GPUs | Data parallelism |
| Cost Reduction | 20-35% | Real-world scenarios |

---

## Method Comparison

| Method | Training Time | Inference | Adaptability | Best For |
|--------|--------------|-----------|--------------|----------|
| **EOQ** | None | < 1ms | Static | Stable demand |
| **Safety Stock** | None | < 1ms | Medium | Service targets |
| **LSTM** | 1-5 min | < 10ms | High | Complex patterns |
| **DQN** | 10-60 min | < 5ms | Very High | Dynamic environments |

---

##  From Research to Production: The End-to-End ML Lifecycle

## Machine Learning System Lifecycle

The JAX Inventory Optimizer follows an end-to-end ML lifecycle — from data collection to deployment and continuous improvement. Future work focuses on automation and scalability, integrating retraining pipelines, distributed JAX infrastructure, feature store consistency, and real-time monitoring to build a fully autonomous optimization engine.

| Stage | Main Objective | Tools / Frameworks | Future Work |
|:------:|----------------|--------------------|--------------|
| **1. Data Preparation (Data)** | Collect, clean, and label structured demand and inventory data | `pandas`, `Airflow`, `SQL` | Automate data ingestion pipelines and implement a centralized feature store. |
| **2. Model Training (Modeling)** | Build, tune, and validate forecasting or control models | `PyTorch`, `JAX`, `scikit-learn` | Extend to distributed JAX training with asynchronous updates. |
| **3. Evaluation & Optimization (Evaluation)** | Compare experiments and select the best-performing model | `Weights & Biases (W&B)`, `MLflow` | Integrate advanced experiment tracking and automated hyperparameter search. |
| **4. Deployment (Deployment)** | Package and deploy as scalable APIs or services | `Docker`, `FastAPI` | Implement load-balanced API endpoints with caching and latency monitoring. |
| **5. Monitoring & Maintenance (MLOps)** | Automate retraining, monitor drift, and manage versions | `Kubernetes`, `Prometheus`, `CI/CD` | Build continuous retraining pipelines with real-time drift detection. |

**Steps 1–3:** make the model *run*  
**Step 4:** make the model *usable in production*  
**Step 5 (MLOps):** make the model *sustain itself and evolve continuously*






<div align="center">
May all our lives keep optimizing — like sleeping better lol. 💤
</div>





