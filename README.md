# 4D-Chaotic-Image-Encryption

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16011486.svg)](https://doi.org/10.5281/zenodo.16011486)

This repository contains the MATLAB implementation of the chaotic image encryption algorithm proposed in our manuscript submitted to *The Visual Computer*. The algorithm integrates chaotic systems with multiple diffusion and permutation stages to achieve high security and performance. Comprehensive experimental evaluations, including statistical, security, and robustness analyses, are included.

📌 **Note:** This code is directly related to the manuscript currently under submission to *The Visual Computer*. If you use this code or data, please cite our corresponding manuscript (citation details below).

---

## 🔍 Overview
This project presents a MATLAB implementation of a novel grayscale image encryption algorithm based on a four-dimensional (4D) continuous chaotic system with strong dissipative properties. The method addresses the limitations of conventional encryption schemes when applied to image data, such as large size and strong inter-pixel correlation.

Key components of the algorithm include:

Image-dependent key generation using SHA-256 to derive initial conditions from the input image

Global Arnold scrambling and intra-block dynamic Zigzag scanning to increase positional randomness

Dual diffusion process combining chaotic orbits, XOR operations, and inter-block chaining for high sensitivity and security

Robust performance evaluation, including metrics such as entropy, NPCR, UACI, and key sensitivity

Simulations on standard test images demonstrate that the algorithm achieves strong security performance, with an average NPCR of 99.61% and UACI of 33.46%, surpassing several state-of-the-art methods.

This code is directly related to our manuscript submitted to The Visual Computer, and is intended to facilitate reproducibility and future research in chaos-based image encryption.



---

## 🚀 Features

- Full image encryption and decryption flow
- Statistical and security analysis (entropy, NPCR, UACI, etc.)
- Key sensitivity test
- MATLAB implementation using `.m` script
- Designed specifically for grayscale images


## 🧪 How to Use

You can use either of the following two methods to obtain and run the code:

🔹 Option 1: Clone this repository (recommended if you use Git)

1.Open a terminal (or Git Bash) and run:

git clone https://github.com/calmbrotheryul/4D-Chaotic-Image-Encryption.git

2.Open MATLAB and set the working directory to the cloned folder.

3.Run the main script:

image_encryption_decryption

🔹 Option 2: Download ZIP or manually copy the code

1.Click the green “Code” button on the top-right of this GitHub page, then choose “Download ZIP”.

2.Extract the ZIP file to a folder.

3.Open MATLAB, set the working directory to the extracted folder.

4.Run the script:

image_encryption_decryption

✅ Note: This project is intended for use with grayscale images only. Make sure your input image is 8-bit grayscale format.


## 🔧 Requirements

MATLAB R2020a or newer


## 📄 Citation
If you use this code or build upon it in your research, please cite the following paper:

Author(s): Yan S., et al.

DOI: [10.5281/zenodo.16011486](https://doi.org/10.5281/zenodo.16011486)

Title: High-Security Image Encryption via a Novel 4D Chaotic System and Dynamic Block Scrambling

Journal: The Visual Computer (under review)

