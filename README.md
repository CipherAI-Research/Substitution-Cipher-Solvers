# Cryptanalysis of Monoalphabetical Substitution Ciphers with Transformer Architectures

[![GitHub License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/Cipher-AI/substitution-cipher-solvers-6731ebd22f0f0d8e0e2e2e00)

Official implementation for the paper "Cryptanalysis of Monoalphabetical Substitution Ciphers with Transformer Architectures". This repository contains code, models, and datasets for solving substitution ciphers using T5 transformer architectures.

---

## 📖 Overview
This work explores transformer-based cryptanalysis of monoalphabetic substitution ciphers using:
1. T5 models fine-tuned to predict plaintext or substitution alphabets
2. Correction models to refine outputs
3. Letter-frequency algorithms for final optimization

Achieves 91.96% accuracy on English texts. [Read the paper](paper_link_placeholder).

---
## Requirements
- transformers
- python-Levenshtein
- datasets
- torch
- pandas

## 🚀 Key Features
- T5 Models: Pre-trained models for ciphertext → plaintext/alphabet prediction
- Correction Pipeline: Multi-step correction system (model + algorithm)
- Benchmark Tools: Evaluate models on custom ciphers
---

## 🧠 Models

### Available Models
| Model Type              | Hugging Face ID                          |
|-------------------------|------------------------------------------|
| Alphabet Prediction     | Cipher-AI/Substitution-Cipher-Alphabet-Eng   |
| Plaintext Prediction    | Cipher-AI/Substitution-Cipher-Text-Eng  |
| Correction Model        | Cipher-AI/AutoCorrect-EN-v2        |

---

## 📊 Results
| Model Configuration                 | Accuracy |
|-------------------------------------|----------|
| Base Alphabet Model                 | 73.56%   |
| + Correction Model                  | 82.81%   |
| + Frequency Algorithm               | 90.87%   |
| Full Pipeline (2 Correction Passes) | 91.96%   |

---

## 📜 Citation
--

---

## 📄 License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

## 📧 Contact
- Şuayp Talha Kocabay: kocabaysuayptalha08@gmail.com
- Kutay Demirbaş: kutay.demirbas@tubitak.gov.tr
