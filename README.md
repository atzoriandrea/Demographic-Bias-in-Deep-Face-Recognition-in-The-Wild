# Demographic Bias in Deep Face Recognition in The Wild

This is the official repository of the paper entitled "Demographic Bias in Deep Face Recognition in The Wild".
<br>
<div align="center">
 <img src="images/overview.png"  width="750" alt="Pipeline Overview"/> 
</div>
<br>

We provide a Pytorch toolbox for Face Images Degradation (1) and Face Recognition training and testing (2). 

1) The Image Degradation module provides 
<br>
<div align="center">
 <img src="images/degradation.jpg"  width="750" alt="Degradation Results"/> 
</div>
<br>
2) The Face Recognition module provides a training part with various SOTA Face Recognition backbones and heads and
an evaluation part that:
- Provides evaluations of the given model(s) in order to obtain metrics like ROC curves, AUCs, EERs, EERs@FAR1% etc.
<br>
<div align="center">
 <img src="./images/ROC.png" height="250" width="250" alt="ROC curve Example"/> <img src="./images/EER.png" height="250" width="250" alt="EER graph Example"/>
</div>
<br>

- Provides metrics as FAR and FRR variation across multiple factors like sex and ethnicity and their combinations
<br>
<div align="center">
 <img src="images/net1.png"  height="130" width="420" alt="Table Example 1"/> <img src="images/net2.png"  height="130" width="420" alt="Table Example 2"/>
</div>
<br>