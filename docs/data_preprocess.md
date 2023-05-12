# Data Preprocessing

## nuScenes

### 1. Preprocessing

To preprocessing the raw data from nuScenes, suppose you have put the raw data of nuScenes at `raw_data_dir`. We provide two modes of proprocessing:
* Only the data on the key frames (2Hz) is extracted, the target location is `data_dir_2hz`.
* All the data (20Hz) is extracted to the location of `data_dir_20hz`.

Run the following commands.

```bash
cd preprocessing/nuscenes_data
bash nuscenes_preprocess.sh ${raw_data_dir} ${data_dir_2hz} ${data_dir_20hz}
```

### 2. Detection

To infer 3D MOT on your detection file, we convert the json format detection files at `file_path` into the .npz files similar to our approach on Waymo Open Dataset. Please name your detection as `name` for future convenience. The preprocessing of the detection follows the below scripts. (Only use `velo` if you want to save the velocity contained in the detection file.)

```bash
cd preprocessing/nuscenes_data

# for 2Hz detection file
python detection.py --raw_data_folder ${raw_data_dir} --data_folder ${data_dir_2hz} --det_name ${name} --file_path ${file_path} --mode 2hz --velo

# for 20Hz detection file
python detection.py --raw_data_folder ${raw_data_dir} --data_folder ${data_dir_20hz} --det_name ${name} --file_path ${file_path} --mode 20hz --velo
```
