# OCR2Text

A deep learning solution designed during the AI4Good hackathon, for the [Meza-OCR-Challenge](https://github.com/Charitable-Analytics-International/AI4Good---Meza-OCR-Challenge). 

The model parses images of hand-written notes composed of digits and a constrained set of punctuations/symbols (comma, dot and minus sign) to text.

The model is basically a resnet stacked on top of a seq2seq model. The idea of using this neural architecture is simple:

- First, use CNN to extract local features to identify singular digits. 
- Then, use seq2seq to combine those local features sequentially for the translation part. Before having known there is something called CTC loss, we had realized that a naive mapping of the feature maps to a translation may not align well.
Therefore, we kindof relied on the attention mechanism in seq2seq model to compensate for the order information.

## Details

The resnet does 3 times down-sampling, and the final features map is flattened into a 1-dimensional representation of the images.

The seq2seq model uses attention on the flattened image feature maps. Beam search can be applied optionally at the inference stage.    
 
## Results 

The general pipeline only has limited efficacy, perhaps due to the limited amount of data, or the limitation imposed by the selection of neural architectures or hyperparams.

Updates: We can achieve 81% as of this point... 

## Data preprocessing and augmentation

We investigated a little bit of image data proprocessing, i.e. segmentation and augmentation.

Image augmentation, which is simply converting images to grey-scale, has added a lot of benefits.
Segmentation on the other hand didn't go very well... 

## Collaborators

Thanks a lot! [Shiquan](https://github.com/zsq007) and [Helen](https://github.com/HelenYou).