# Self-SSGI
Self-SSGI is a multimodal self-supervised learning model that integrates protein sequences, structures, GO annotations, and images to learn high-resolution protein representations.

## Data Preprocessing

### 1. Protein Atom-Level Feature Extraction:
- Input PDB files and run `Process_Seq_Struc_data.m` to obtain amino acid types and dihedral features.
- Use [mini3di](https://github.com/althonos/mini3di) to obtain Foldseek features.

#### Feature Order:
- **1-10**: Foldseek features.
- **11-12**: Dihedral angles(φ, ψ).
- **13-15**: The X, Y, and Z coordinates of the alpha carbon (CA) atoms of the amino acids.
- **16-35**: One-hot encoding of amino acid types.

#### Normalization:
- When processing amino acid level features, values are normalized to ensure they fall within the same range.
- **Foldseek_Min**: `[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]`
- **Foldseek_Max**: `[2.9393, 5.1267, 4.4613, 5.0133, 4.5739, 2.5407, 3.9567, 2.6599, 4.8151,2.6291]`
- **11-15_Min**: `[-179.9999, -179.9998, -259.9510, -147.0740, -246.7790]`
- **11-15_Max**: `[179.9997, 179.9999, 293.0810, 141.9500, 247.6480]`
- normalized_features = (features - min_values) / (max_values - min_values)

### 2. Single-cell Image Processing:
- Place 512x512x4 single-cell `.mat` images in `Image_Data` directory. Each image should be composed of four channels in the exact RGBY order: Red (grayscale image of Microtubules), Green (grayscale image of the Target Protein), Blue (grayscale image of the Nucleus), and Yellow (grayscale image of the Endoplasmic Reticulum, ER).
- The example is already in the Image_Data folder.
  
## Self-supervised model for joint representation learning of protein sequence and structure based on mask training
- Run train.py to train the model. You can download the pre-trained model at [Pretrained Sequence_Structure Model](https://huggingface.co/Maureen123/Self-SSGI/blob/main/Self-SSGI_Pretrained%20Sequence_Structure%20Model.pkl)
## Contrastive Image-GO Pre-training
- Run load_model_and_get_fea.py to  load our pre-trained model and extract the features of the image. You can download the pre-trained model at [Pretrained Image Model](https://huggingface.co/Maureen123/Self-SSGI/blob/main/Self-SSGI_Pretrained%20Image%20Model.pkl)
  
## Multimodal data fusion
- Run train.py to train the model, and the input data for this part requires that features from different modalities are unified to a dimension of 128. 
