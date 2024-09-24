<h1>Pneumonia Prediction from Chest X-ray Images</h1>

<p>This repository contains the implementation of pneumonia prediction from chest X-ray images using various machine learning and deep learning techniques, including Transfer Learning models such as VGG16 and ResNet50.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#methodology">Methodology</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
  <li><a href="#future-work">Future Work</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>Pneumonia is a respiratory infection that poses a significant public health burden, leading to millions of hospitalizations globally. This project applies deep learning and machine learning techniques to detect pneumonia using chest X-ray images.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used consists of chest X-ray images divided into two classes: Normal and Pneumonia. The dataset was sourced from publicly available resources and contains around 6000 images in each class.</p>

<h2 id="methodology">Methodology</h2>
<p>We compare different models, including traditional Machine Learning models (SVM, Random Forest) and Deep Learning models (VGG16, ResNet50), to evaluate their performance in predicting pneumonia. Transfer Learning is utilized to fine-tune pre-trained models for improved accuracy.</p>

<h2 id="model-architecture">Model Architecture</h2>
<p>The architecture consists of convolutional layers for feature extraction, followed by dense layers for classification. Transfer Learning models like VGG16 and ResNet50 are fine-tuned for pneumonia prediction.</p>

<h2 id="results">Results</h2>
<p>The VGG16 model achieved the best accuracy of 88.86%, outperforming other models. Below are the accuracy results for the top models:</p>
<ul>
  <li>VGG16: 88.86%</li>
  <li>ResNet50: 84.20%</li>
  <li>CNN: 70.00%</li>
</ul>

<h2 id="conclusion">Conclusion</h2>
<p>The VGG16 model emerged as the best performer in predicting pneumonia from chest X-ray images, showcasing the effectiveness of Transfer Learning in medical image analysis.</p>

<h2 id="future-work">Future Work</h2>
<p>Future work includes improving multi-class classification and exploring attention mechanisms for better feature extraction.</p>

<h2>Keywords</h2>
<ul>
  <li>VGG16</li>
  <li>ResNet50</li>
  <li>CNN</li>
</ul>

<h2>References</h2>
<ul>
  <li><a href="https://doi.org/10.1109/EBBT.2019.8741582">Diagnosis of Pneumonia from Chest X-Ray Images Using Deep Learning</a></li>
  <li><a href="https://doi.org/10.1371/journal.pone.0256630">Pneumonia detection in chest X-ray images using an ensemble of deep learning models</a></li>
</ul>
