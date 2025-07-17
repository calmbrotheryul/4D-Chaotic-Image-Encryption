# 4D-Chaotic-Image-Encryption
This repository contains the MATLAB implementation of the chaotic image encryption algorithm proposed in our manuscript submitted to *The Visual Computer*. The algorithm integrates chaotic systems with multiple diffusion and permutation stages to achieve high security and performance. Comprehensive experimental evaluations, including statistical, security, and robustness analyses, are included.

📌 **Note:** This code is directly related to the manuscript currently under submission to *The Visual Computer*. If you use this code or data, please cite our corresponding manuscript (citation details below).

---

## 🔍 Overview

This project provides a full MATLAB implementation of a chaotic image encryption scheme that leverages nonlinear dynamic behavior to ensure secure image transmission. The encryption pipeline includes:

- Chaotic sequence generation
- Image permutation and diffusion
- Performance evaluation (histogram, entropy, NPCR, UACI, key sensitivity, etc.)

The algorithm is designed to be **reproducible** and **easy to adapt** for future research on chaos-based cryptography.

---

## 🚀 Features

- Full image encryption and decryption flow
- Statistical and security analysis (entropy, NPCR, UACI, etc.)
- Key sensitivity test
- MATLAB implementation using `.m` script
- Designed specifically for grayscale images

---

## 📁 File Structure

```text
 ┣ 📄 image_encryption_decryption.m          # Main script containing the full pipeline
 ┣ 📄 README.md                              # Documentation
 ┣ 📁 example_images/                        # Grayscale test images used in the paper
 ┣ 📁 results/                               # Sample encrypted/decrypted images and metrics
 

🧪 How to Use

1.Clone this repository

git clone https://github.com/your-username/ChaoticImageEncryption.git
Open MATLAB and set the directory to the cloned folder.

2.Run the main script

image_encryption_decryption

3.Adjust Parameters

You may customize image paths and chaotic system parameters in image_encryption_decryption.m.

🔧 Requirements
MATLAB R2020a or newer


📄 Citation
If you use this code or build upon it in your research, please cite the following paper:

Author(s): [Your Name], et al.
Title: [Title of your manuscript]
Journal: The Visual Computer (under review)
DOI: [DOI link once available]

Additionally, please cite this GitHub repository or its Zenodo DOI (if uploaded there).
