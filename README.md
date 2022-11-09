Cataract surgical videos analysis based on cross domain data

Introduction:

Surgical picture semantic segmentation is attracting increasing attention from the medical image processing community. The goal generally is not to precisely locate tools in images, but rather to indicate which tools are being used by the surgeon at each instant. The main motivation for annotating tool usage is to design efficient solutions for surgical workflow analysis. Analyzing the surgical workflow has potential applications in report generation, surgical training, and even real-time decision support. 

We propose an innovative framework on cross-domain data in the clinical application of the cataract surgical tool segmentation. We first go through a frequency space domain randomization methods that transforms cataract surgery images into frequency space and performs domain generalization by identifying and randomizing domain-variant frequency components (DVFs) while keeping domain invariant frequency components (DIFs) unchanged. Based on the DIFs and DVFs, we apply domain Randomization. After that with multi-view methods, shared features of both domains are preserved and therefore improve the performance of semantic segmentation. Then we apply a category-level adversarial network to enforce local semantic consistency during the trend of global alignment.
