# INSPIRE
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Abstract
Hierarchical neural networks are essential for multi-scale representation in medical imaging, facilitating tasks like object detection and segmentation. Yet, their performance is dampened by intra-scale information loss and inter-scale feature misalignment. To address these challenges, We propose an INtegrated-Scale Pyramidal Interactive Reconfiguration to Enhance feature learning (INSPIRE). This approach is designed to enrich intra-scale semantic enhancement, ensure precise inter-scale spatial alignment, and is coupled with spatial-semantic back augmentation. Our method outshines standard baselines in diverse medical segmentation challenges, notably in identifying brain tumors, stroke lesions, and polyps. We validate the generalizability of INSPIRE by extending it to object detection and semantic segmentation in natural images. INSPIRE boosts semantic clarity, aligns multi-scale features, and achieves integrated spatial-semantic coherence. It seamlessly meshes with widely employed hierarchical neural networks in medical image analysis, such as U-Net and FPN, promising to enhance computer-aided diagnostics and clinical interventions. 

![Architecuture](figs/arch.png)

## Installation
This implementation is built on the[mmdetection](https://github.com/open-mmlab/mmdetection). Please follow the guildlines of [mmdetection](https://github.com/open-mmlab/mmdetection) to install.

## Training

We use the slurm system to train our model. [Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.

On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows.

```shell
OMP_NUM_THREADS=1 [GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/fpn_htc_r50.py ${WORK_DIR}
```

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--options`. This is more recommended since it does not change the original configs.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   In `config2.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

## Testing

The test script is similar to the train script. 

```shell
OMP_NUM_THREADS=1 [GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} configs/fpn_htc_r50.py ${WORK_DIR}
```