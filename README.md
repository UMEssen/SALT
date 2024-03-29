# Softmax for Arbitrary Label Trees (SALT)

This segmentation framework was developed at the Institute for AI in Medicine of the University Hospital Essen by the [SHIP-AI team](https://ship-ai.ikim.nrw/).
The framework can be used for any 2D or 3D segmentation task that exhibits a hierarchical labels structure.
In our case, we applied this to medical imaging and provide the segmentation of 145 different structures in the human body.

The model is very fast (average of 35 seconds for a 1000 whole body!) and can segment 113 body regions in a single pass.

![Segmentation](https://github.com/UMEssen/SALT/blob/main/images/segmentation.gif)

## Training

### Datasets

For training, the dataset folder should have the following structure:

```
data/
├── lits/
├── kits/
├── saros/
├── ...
```

Each dataset should contain the following files and folders:

```
data/
├── saros/
│   ├── labels.txt
|   ├── tree-labels.txt
│   ├── train/
│   │   ├── images/
│   │   │   ├── s0001.nii.gz
│   │   │   ├── s0002.nii.gz
│   │   ├── labels/
│   │   │   ├── s0001.nii.gz
│   │   │   ├── s0002.nii.gz
│   ├── val/
│   │   ├── images
│   │   │   ├── ...
│   │   ├── labels
│   │   │   ├── ...
│   ├── test/
│   │   ├── images
│   │   │   ├── ...
│   │   ├── labels
```

The datasets do not need to have all three `train`, `val`, and `test` folders, and may only include a `test` set.

The files `labels.txt` and `tree-labels.txt` specify the labels from the dataset. `labels.txt` should contain the label names, while `tree-labels.txt` should define the hierarchical structure of the labels from the tree. In the [labels](./labels) folder, you can find some examples for these files, and in the [conversion](salt/data/conversion) you can find examples on how the data was converted to this format.


### Train
Once the data is set up, you can build the docker image with
```bash
docker build -t shipai/salt .
```

and then run the container with
```bash
docker run -it --rm \
       --runtime=nvidia \
       --network host \
       --user $(id -u):$(id -g) \
       --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /path/to/storage:/storage \
       -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
       shipai/salt
```

Once the container is running, you can use the following commands to train the models:
```bash
python3 -m salt.train \
    --data-dir /storage/datasets/data/ \
    --mixed-precision \
    --train-dir /storage/models/your-model-name \
    --epochs 1000 --batch-size 3 --gpus 0 1 2 --sink-weight 0.0
```

The number of gpus and the batch size you use depends on how many GPUs you made available in your docker run command.

Once the training is finished, you can export the model to a single model file:
```bash
python -m salt.export \
       --train-dir /storage/models/your-model-name \
       --output-dir /storage/models/exports/your-model-name
```

## Prediction & Evaluation

You can build the docker image with
```bash
docker build -t shipai/salt .
```

and then run the container with
```bash
docker run -it --rm \
       --runtime=nvidia \
       --network host \
       --user $(id -u):$(id -g) \
       --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
       -v /path/to/storage:/storage \
       -e NVIDIA_VISIBLE_DEVICES=0,1,2 \
       shipai/salt
```

Once the container is running, you can compute the predictions for your dataset using our model
```bash
python -m salt.predict \
       --data-dir /storage/datasets/data/dataset-name/images \
       --output_dir /storage/results/your-model-name/dataset-name
```

or run it with your newly trained model

```bash
python -m salt.predict \
       --model-file /storage/models/exports/your-model-name/model.pt \
       --config-file /storage/models/exports/your-model-name/config.pkl \
       --data-dir /storage/datasets/data/dataset-name/images \
       --output_dir /storage/results/your-model-name/dataset-name
```

Note that for the predictions, the data does not need to be in a particular format, and any folder containing NIfTI files can be used as input.

If your dataset has a ground truth, you can evaluate the predictions from outside the container using:

```bash
poetry run python -m salt.classic_evaluate \
       --config-file /path/to/the/model/config.pkl \
       --data-dir /path/to/the/data \
       --predictions-dir /path/to/the/predictions \
       --output-dir /path/for/results
```
