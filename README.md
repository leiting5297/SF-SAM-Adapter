# SF-SAM-Adapter

SF-SAM-Adapter, a advanced segmentation method based on the observation of the spatial and frequency features of the segmented target, integrates the prior knowledge to the large segmentation model [SAM](https://github.com/facebookresearch/segment-anything), by simple yet effective Adapter technique [Adaption](https://lightning.ai/pages/community/tutorial/lora-llm/).


## Requirement

``conda env create -f environment.yml``

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

## Example

1. The dataset is placed in "./data/eye", and the file structure is:

EYE/

     EYE_Test_Data/...
     
     EYE_Training_Data/...
     
     EYE_Test_GroundTruth.csv
     
     EYE_Training_GroundTruth.csv
    
2. Train: ``python train.py -net sam -mod sam_adpt -exp_name *eye* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset eye --data_path *../data*``
 

3. Evaluation: The code can automatically evaluate on the test set during traing, and you can also manually evaluate it by running val.py for.


Results will be saved at `` ./logs/`` in default.



