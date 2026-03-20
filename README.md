# 🧠 Customer Segmentation using Autoencoders and Clustering

## 🧠 Overview

This project implements an advanced customer segmentation pipeline using Autoencoders for non-linear dimensionality reduction followed by K-Means clustering. The goal is to uncover meaningful customer groups from high-dimensional sales data by learning compact latent representations.

---

## 🎯 Objective

- Reduce high-dimensional data into a compact latent space  
- Learn meaningful feature representations using an autoencoder  
- Perform clustering on learned representations  
- Improve cluster separability compared to raw feature space  

---

## 🏗️ Methodology

### Data Preparation

- Input dataset: sales_df  
- Features are scaled using normalization techniques (e.g., StandardScaler)  

Checking `sales_df.shape` ensures:
- Correct input dimension for the autoencoder  
- Each feature corresponds to one input neuron  

---

## 🤖 Autoencoder Architecture

The autoencoder is designed to reconstruct input data while learning compressed representations.

### Structure

Encoder:
- Dense layers reduce dimensionality  
- Activation: ReLU (introduces non-linearity)  
- Initialization: glorot_uniform (stable weight initialization)  

Bottleneck Layer:
- Compressed representation of input  
- Captures most important features  

Decoder:
- Reconstructs input from compressed representation  

---

## ⚙️ Training Setup

- Optimizer: Adam  
- Loss Function: Mean Squared Error  

Training is done as:

Input = sales_df  
Target = sales_df  

The model learns to reconstruct its input, forcing compression through the bottleneck layer.

### Parameters

- Batch size → number of samples per update  
- Epochs → number of training iterations  
- Verbose → controls training output display  

---

## 💾 Saving Model

Model weights are saved using:

autoencoder.save_weights(...)

Purpose:
- Preserve learned knowledge  
- Avoid retraining  
- Enable reuse  

---

## 🔽 Dimensionality Reduction

Encoded features are generated using:

pred = encoder.predict(sales_df_scaled)

Result:
- pred contains lower-dimensional representations  
- Captures important patterns from original data  

---

## 📉 Elbow Method

The elbow method is applied on pred to determine optimal clusters.

Purpose:
- Identify best number of clusters  
- Evaluate clustering in latent space  

---

## 🔗 K-Means Clustering

K-Means is applied on encoded features:

kmeans.fit(pred)

Output:
- Cluster labels for each data point  

---

## 📊 Cluster Assignment

Cluster labels are added to dataset:

df_cluster_dr = data + cluster_labels

Each data point is now assigned a cluster.

---

## 📍 Cluster Centers (Scaled)

Cluster centers represent:
- Mean values of each cluster in scaled space  

These are not directly interpretable due to scaling.

---

## 🔄 Inverse Transformation

Using:

scaler.inverse_transform(cluster_centers)

This converts cluster centers back to original scale for interpretation.

---

## 🧠 Cluster Interpretation

Cluster 1:
- High value / high engagement customers  

Cluster 2:
- Moderate activity customers  

Cluster 3:
- Low engagement customers  

---

## 📊 Visualization

### Histograms

Feature-wise histograms are plotted for each cluster to analyze distributions.

---

### 3D Scatter Plot

- Axes: PCA components  
- Color: Cluster label  
- Used to visually inspect cluster separation  

---

## 🔬 Observations

- Autoencoder improves clustering quality  
- Latent space captures hidden patterns  
- Clusters are more separable than raw data  

---

## 💡 Key Contributions

- Applied autoencoder for dimensionality reduction  
- Combined deep learning with clustering  
- Enabled meaningful customer segmentation  
- Visualized clusters using PCA  

---

## ⚠️ Limitations

- Sensitive to architecture design  
- Requires tuning of bottleneck size  
- K-Means assumes spherical clusters  
- Interpretation depends on dataset  

---

## 🚀 Future Work

- Use Variational Autoencoders (VAE)  
- Try DBSCAN or Gaussian Mixture Models  
- Add temporal features  
- Improve interpretability  

---

## 📁 Project Structure

Autoencoder_Clustering/
├── notebook.ipynb  
├── models/  
├── outputs/  

---

## 📦 Requirements

pip install tensorflow keras scikit-learn numpy pandas matplotlib  

---

## ✍️ Author

Sai Sohan Sajja  
Machine Learning | Deep Learning | Applied AI  

---

## 🧠 Summary

This project demonstrates how autoencoders can learn compact, non-linear representations that significantly improve clustering performance, enabling better segmentation of high-dimensional data.
