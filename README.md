# Self-SSGI
Self-SSGI is a multimodal self-supervised learning model that integrates protein sequences, structures, GO annotations, and images to learn high-resolution protein representations.

## Data Preprocessing

### 1. Protein Feature Extraction:
- Input PDB files and run `Process_Seq_Struc_data.m` for amino acid types and dihedral angles.
- Use `mini3di` for Foldseek encoding.

#### Feature Order:
- **1-10**: Foldseek
- **11-12**: Dihedral angles
- **13-15**: X, Y, Z coordinates
- **16-35**: One-hot amino acid encoding

#### Normalization:
- **Min**: `[-179.9999, -179.9998, -259.9510, -147.0740, -246.7790, ..., 0.0]`
- **Max**: `[179.9997, 179.9999, 293.0810, 141.9500, 247.6480, ..., 4.8151]`

### 2. Single-cell Image Processing:
- Place 512x512 single-cell `.mat` images in `Image_Data` with RGBy channels for grayscale red, green, blue, and yellow images.
