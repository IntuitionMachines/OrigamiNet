# OrigamiNet

Public implementation of our CVPR 2020 paper:

"OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold"

<p align="center">
<img src="./o1.png">
</p>

## Getting Started

OrigamiNet has been implemented and tested with Python 3.6 and PyTorch 1.3. All project configuration is handled using [Gin](https://github.com/google/gin-config).

First clone the repo:
```
git clone https://github.com/IntuitionMachines/OrigamiNet.git
```
Then install the dependencies with:
```
pip install -r requirements.txt
```
## Replicating Experiments

### IAM
1. Register at the FKI's webpage [here](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php).

2. After obtaining the username and password, we provide a script to download and setup the dataset, crop paragraph images and generate corresponding paragraph transcriptions by concatenating each line transcription. Run:
```
bash iam/iam.sh $IAM_USER $IAM_PASS $IAM_DEST
```
where `$IAM_USER` and `$IAM_PASS` are the username and password from FKI website, `IAM_DEST` is the destination folder where the dataset will be saved (the folder will be created by the script if it doesn't exist).

3. Run the training script using provided configuration:
```
python train.py --gin iam/iam.gin
```
Note: if you want to use `horovod`, run as following:
```
horovodrun -n $N_GPU -H localhost:$N_GPU python train.py --gin iam/iam.gin
```
where `$N_GPU` is the number of gpus to be used (visible GPUs can be controlled by setting `CUDA_VISIBLE_DEVICES`)

### ICDAR2017 HTR

1. Download and set up the dataset using the provided script:
```
bash ich17/ich.sh $ICH_DEST
```
`ICH_DEST` is the destination folder where the dataset will be saved. The folder will be created by the script if it doesn't exist.

2. Run the training script using provided configuration:
```
python train.py --gin ich17/ich.gin
```

## Results

In the following table CER and nCER are respectively the micro and macro averaged [Character Error Rate](https://en.wikipedia.org/wiki/Levenshtein_distance). BLEU is the marco-averaged character-level [BLEU](https://en.wikipedia.org/wiki/BLEU) score.

### Paper results

Dataset | wmul | Size    | CER (%)  | nCER (%) | BLEU
--------|------|---------|--------- | -------- |-----
IAM     | 1.5  | 750x750 |  4.7     | 4.84     | 91.15
ICDAR   | 1.8  |1400x1000|  6.80    | 5.87     | 92.67


### Additional results

Dataset | wmul | Size    | CER (%)  | nCER (%) | BLEU
--------|------|---------|--------- | -------- |-----
IAM     | 1.0  | 750x750 |  4.85    | 4.95     | 90.87
IAM     | 2.0  | 750x750 |  4.41    | 4.54     | 91.25
IAM     | 3.0  | 750x750 |  4.29    | 4.41     | 91.84
IAM     | 4.0  | 750x750 |  4.07    | 4.18     | 92.21
ICDAR   | 2.4  |1400x1000|  6.01    | 5.30     | 93.64

These experiments were done with a `batch_size` of 8. We also obtained promising results with a `batch_size` of 4, as the proposed architecure does not utilize BatchNorm operations.

### Synthetic hard-to-segment IAM variants

In the paper, two IAM variants with hard-to-segment text-lines were presented. These results can be replicated as follows:

#### Compact lines

1. Make a copy of the `pargs` folder, which contains the extracted paragraph images:
```
cp -r iam/pargs/ iam/pargsCL
```
2. To generate IAM with touching lines, use `image-magick` to resize images to half the height using seam carving.

The following line runs the conversion in parallel to speed up the process:
```
find iam/pargsCL -iname "*.png" -type f -print0 | parallel --progress -0 -j +0 "mogrify -liquid-rescale 100x50%\! {}"
```

#### Rotated and warped

1. Make a copy of the `pargs` folder, which contains the extracted paragraph images:
```
cp -r iam/pargs/ iam/pargsPW
```
2. To generate IAM with a random projection and wavy text-lines:

```
find iam/pargsPW -iname "*.png" -type f -print0 | parallel --progress -0 -j +0 "python dist.py {}"
```
### Results

Dataset            | wmul | Size    | CER (%)  
-------------------|------|---------|--------- 
Compact lines      | 1.0  | 750x750 |  6.0    
Rotated and warped | 1.0  | 750x750 |  5.6    

## Single line results

To be as useful as possible, we show how to perform single-line recognition based on the code. This essentially resembles the [GTR model](https://arxiv.org/abs/1812.11894). Assuming lines from IAM and thier transcriptions are stored in `iam/lines/`, run as
```
python train.py --gin iam/iam_ln.gin
```
### Results

Results on the IAM single-line test set

Dataset        | nlyrs | Size    | CER (%)  
---------------|-------|---------|--------- 
IAM lines      | 12    | 32x600  |  5.26    
IAM lines      | 18    | 32x600  |  4.84    
IAM lines      | 24    | 32x600  |  4.76     



## Gin Options

This is a brief list of the most important gin options. For full config files see `iam/iam.gin` or `ich17/ich.gin`
- `dist`: The parallel traning method. We currently support three possible values: 
  - `DP` uses DataParallel
  - `DDP` uses DistributedDataParallel
  - `HVD` uses horovod
- `n_channels`: number of channels per image
- `o_classes`: The size of the target vocabulary (i.e. number of symbols in the alphabet)
- `GradCheck`: Whether or not to use gradient checkpointing
  - `0` disabled
  - `1` enabled, light version which offers good memory saving with small slowdown
  - `2` enabled, higher memory saving, but noticeably slower than `1`
- `get_images.max_h` and `get_images.max_w`: Target height and width for each image, images will be resized to this target dimentions while maintaining aspect ratio by padding.
- `train.AMP`: Whether Automatic Mixed Precision (by [Nvidia apex](https://github.com/NVIDIA/apex)) is enabled
- `train.WdB`: Whether [Wandb](https://github.com/wandb/client) logging is enabled
- `train.train_data_list` and `train.test_data_list`: Path to file containing list of training or testing images
- `train.train_data_path` and `train.test_data_path`: Path to folder containing the training or testing images
- `train.train_batch_size` and `train.val_batch_size`: the batch size used during training and validation, the interpretation of this value vaires according to `dist` option
  - `DP` the `train.batch_size` is the total batch size
  - `DDP` or `HVD` then `train.batch_size` is the batch size per process (total batch size is `train.batch_size*#Processes`
- `train.workers`: Number of worker for the PyTorch `DataLoader`
- `train.continue_model`: Path to checkpoint to continue from
- `train.num_iter`: Total number of training iterations
- `train.valInterval`: Perform validation every how many batches
- `OrigamiNet.nlyrs`: #layers in the GTR model
- `OrigamiNet.reduceAxis`: Final axis of reduction
- `OrigamiNet.wmul`: Channel multipler, numer of channels in each channel will be multiplied by this value
- `OrigamiNet.lszs`: Number of channels for each layer, this is a dictionary of format `leyer_id:channels`, unspecified layers are assumed constant
- `s1/Upsample.size`: Size of penultimate layer
- `s1/Upsample.size`: Size of the last layer
- `OrigamiNet.lreszs`: Resampling stages in the model

## Acknowledgements

Some code is borrowed from the [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark),
which is under the Apache 2.0 license.

Network architecture was visualized using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

This work was sponsored by [Intuition Machines, Inc](https://www.imachines.com/).

## Citation

```bibtex
@inproceedings{yousef2020origaminet,
  title={OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page TextRecognition by learning to unfold},
  author={Yousef, Mohamed and Bishop, Tom E.},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
