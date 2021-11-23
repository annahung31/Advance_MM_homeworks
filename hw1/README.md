
## Environment  
1. Create conda environment  
```
conda create -n r08922a20 python=3.7.10
```
2. Activate environment  
```
conda activate r08922a20
```
3. Install requirements 
```
pip install -r requirements.txt
```
4. Install PyTorch  
If your CUDA verison is 10.1:   
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
Otherwise please refer to [here](https://pytorch.org/get-started/previous-versions/) to find the version compatible for your CUDA.

## Usage  

### Training
1. Change the experiment folder name **config.output** in `config.py`. The model will be saved to it.
2. Run script:  

```
python main.py --mode train
```

### Testing  
For closed set:  
```
python main.py --mode closed
```

For open set:  
```
python main.py --mode open
```

* If this is not the first time for you to run closed set testing or open set testing, you can skip the data preprocessing to save time:  
```
python main.py --mode closed --data_process 0
```


## Data preparation Notes
### For training ans validation

1. align the images.  The folder strcuture needs to be `-root/personA/*.img.` 
```
python align_dataset_mtcnn_v1.py ../test/closed_set/test_pairs/ ../test/closed_set/test_pairs_align
```
2. generate landmark:  
```
python get_bbox_lnmk.py
```
3. generate .lst file:   

```
python insightface_pairs_gen_v1.py --dataset-dir  --list-file
```

4. Generate .rec and .idx files using: face2rec2.py   




## Reference
[1] [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)   
[2] [How to prepare data](https://github.com/deepinsight/insightface/issues/791)     
[3] [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model)   
