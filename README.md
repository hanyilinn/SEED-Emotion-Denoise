# SEED-Emotion-Denoise

This repository contains the code for **ASTI-Net**, a model designed for **EEG denoising**. In this work, we perform experiments to evaluate the performance of **TSception** and **ShallowConvNet** classifiers both with and without the application of **ASTI-Net** denoising on the **SEED** dataset. The goal of the experiments is to investigate how denoising impacts the classification accuracy of EEG-based emotion recognition tasks.

## Dataset
The **SEED dataset** is used for this study, a popular EEG dataset for emotion recognition tasks. The dataset contains EEG recordings of subjects while watching emotional video clips. The aim is to classify the emotions based on the EEG signals, both before and after denoising.

## Models
- **ASTI-Net**: The proposed model for EEG denoising, which cleans the EEG signal before feeding it into classification models. This approach aims to reduce noise and improve classification performance.
- **TSception**: A deep learning architecture designed for capturing temporal dynamics and spatial asymmetry in EEG signals. This model is discussed in the paper [2].
- **ShallowConvNet**: A traditional CNN-based model used for EEG signal classification.

The experiments compare the performance of these classifiers both with and without the application of **ASTI-Net** denoising to evaluate how denoising improves classification accuracy.

## Requirements
To run the experiments, ensure that you have the following dependencies installed:
- Python 3.x
- PyTorch
- NumPy
- SciPy

## References
1. **Zheng, Wei-Long, and Bao-Liang Lu**. "Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks." *IEEE Transactions on Autonomous Mental Development*, 7.3 (2015): 162-175. [Link to paper](https://ieeexplore.ieee.org/document/7169937)
2. **Ding, Yi, et al.** "TSception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition." *IEEE Transactions on Affective Computing*, 14.3 (2022): 2238-2250. [Link to paper](https://ieeexplore.ieee.org/document/9751782)
3. **Schirrmeister, Robin Tibor, et al.** "Deep learning with convolutional neural networks for EEG decoding and visualization." *Human Brain Mapping*, 38.11 (2017): 5391-5420. [Link to paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23706)
