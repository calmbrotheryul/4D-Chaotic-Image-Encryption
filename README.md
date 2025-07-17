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

Author(s): [Your Name], et al.
Title: [Title of your manuscript]
Journal: The Visual Computer (under review)
DOI: [DOI link once available]

Additionally, please cite this GitHub repository or its Zenodo DOI (if uploaded there).

---
