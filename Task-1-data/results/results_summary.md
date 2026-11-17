# Task 1: Google Speech Commands v0.2 — Results Summary

This document collects all required values, specifications, and benchmark metrics for Tasks 1.2, 1.3, and 1.4.

---

## 1. System Specifications (PC Specs)

* **CPU**: AMD Ryzen 9 5950X (16 Cores / 32 Threads @ up to 4.9 GHz)
* **GPU**: NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM)
* **RAM**: 32 GB DDR4
* **OS**: Ubuntu 26.04.1 LTS (Linux kernel 6.14.0)
* **Python Environment**: Python 3.12, PyTorch 2.15.0+cu13, Brevitas 0.13.4, QONNX 1.0.0

---

## 2. Task 1.2: Standard Floating-Point Training (FP32)

* **Architecture**: 4-Layer 2D Convolutional Network (`Conv2D` + `BatchNorm2D` + `ReLU` + `MaxPool2D` + `Dropout` + `Linear`) on `(1, 10, 49)` MFCC features.
* **Training Command**:
  ```bash
  python model_float32.py kws_float32 --train --epochs 15 --lr 0.001 --bsize 256
  ```

### Required Metrics:
* **Item 4**:
  * **Model size**: 1.8 MB (`models/kws_float32.pt` / `models/kws_float32.onnx`)
  * **Time**: 34.79 s
  * **Validation Accuracy**: 96.86%
* **Item 5**:
  * **Accuracy (Test)**: 96.30%
  * **Total BOPS**: 11,969,036,288 ($1.197 \times 10^{10}$)
  * **Total Weight Bits**: 14,746,624 bits (~1,800 KB)
* **Item 6 (Latency on 10,083 test samples)**:
  * **Avg. Time**: 0.0199 s (19.9 ms)
  * **Min. Time**: 0.0171 s (17.1 ms)
  * **Max. Time**: 0.0621 s (62.1 ms)

---

## 3. Task 1.3: Quantization-Aware Training (QAT)

* **Architecture**: 4-Layer MLP with Brevitas `QuantLinear`, `BatchNorm1d`, and `QuantReLU` layers.
* **Training Commands**:
  ```bash
  python model_quant.py kws_w8a8 --train --epochs 10 --lr 0.005 --bsize 256
  python model_quant.py kws_w4a4 --train --epochs 10 --lr 0.005 --bsize 256
  python model_quant.py kws_w2a2 --train --epochs 10 --lr 0.005 --bsize 256
  ```

### Item 4: Training & Validation Metrics for each Level of Quantization
1. **Level 1 (W8A8 — 8-bit Weights, 8-bit Activations)**:
   * **Model size**: 926 KB (`.pt`) / 940 KB (`.onnx`)
   * **Time**: 39.54 s
   * **Validation Accuracy**: 89.84%
2. **Level 2 (W4A4 — 4-bit Weights, 4-bit Activations)**:
   * **Model size**: 926 KB (`.pt`) / 940 KB (`.onnx`)
   * **Time**: 39.59 s
   * **Validation Accuracy**: 89.62%
3. **Level 3 (W2A2 — 2-bit Weights, 2-bit Activations)**:
   * **Model size**: 926 KB (`.pt`) / 940 KB (`.onnx`)
   * **Time**: 38.94 s
   * **Validation Accuracy**: 85.78%

### Item 5: Test Accuracy & QONNX Inference Cost
1. **Level 1 (W8A8)**:
   * **Accuracy**: 87.66%
   * **Total BOPS**: 14,409,728 ($1.44 \times 10^7$)
   * **Total Weight Bits**: 1,801,216 bits (~220 KB)
2. **Level 2 (W4A4)**:
   * **Accuracy**: 87.44%
   * **Total BOPS**: 5,609,472 ($5.61 \times 10^6$)
   * **Total Weight Bits**: 900,608 bits (~110 KB)
3. **Level 3 (W2A2)**:
   * **Accuracy**: 84.49%
   * **Total BOPS**: 2,405,888 ($2.41 \times 10^6$)
   * **Total Weight Bits**: 450,304 bits (~55 KB)

### Item 6: Latency Benchmark for Best Quantized Model (`kws_w8a8.onnx` on 10,083 samples)
* **Avg. Time**: 0.0201 s (20.1 ms)
* **Min. Time**: 0.0167 s (16.7 ms)
* **Max. Time**: 0.0440 s (44.0 ms)

---

## 4. Task 1.4: Cross-Model Comparison Table

| Model | Topology | Precision (W/A) | Val Acc (%) | Test Acc (%) | Total BOPS | BOPS Reduction vs FP32 | Total Weight Bits | Weight Memory | Avg CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32** | Conv2D | 32 / 32 | 96.86% | **96.30%** | 11,969,036,288 | 1.0x | 14,746,624 | ~1,800 KB | 19.9 ms |
| **W8A8** | MLP | 8 / 8 | 89.84% | **87.66%** | 14,409,728 | **830x** | 1,801,216 | ~220 KB | 20.1 ms |
| **W4A4** | MLP | 4 / 4 | 89.62% | **87.44%** | 5,609,472 | **2,133x** | 900,608 | ~110 KB | 19.5 ms |
| **W2A2** | MLP | 2 / 2 | 85.78% | **84.49%** | 2,405,888 | **4,975x** | 450,304 | ~55 KB | 19.5 ms |

---

## 5. Result Plots in this Directory

* `results/model_comparison.png`: Comprehensive 4-panel comparison bar chart.
* `results/kws_float32_accuracy.png`: Training & validation curves for FP32 model.
* `results/kws_w8a8_accuracy.png`: Training & validation curves for W8A8 QNN.
* `results/kws_w4a4_accuracy.png`: Training & validation curves for W4A4 QNN.
* `results/kws_w2a2_accuracy.png`: Training & validation curves for W2A2 QNN.
