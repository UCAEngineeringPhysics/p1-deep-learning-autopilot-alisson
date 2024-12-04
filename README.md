[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZBaC74ep)
# Final Project - Deep Learning Autopilot
## Objectives
- Develop a deep learning autopilot model based on convolutional neural networks.
- Use behaviroal cloning approach to train this autopilot.

## Usage
1. Clone this repository to Raspberry Pi (and rename it to dlr).
```bash
cd ~
git clone https://github.com/<team_name>/<repository_name> dlr
```
2. Collect data
```bash
python ~/dlr/scripts/collect_data.py
```
3. Transfer data
```bash
rsync -rv --partial --progress ~/dlr/data/2024-11-12-13-14 user@192.168.0.112:~/dlr/data/
```
4. Log in to server
```bash
ssh user@192.168.0.112
```
5. Train model
```bash
mamba activate bc
python ~/dlr/scripts/train.py 2024-11-12-13-14
```
6. Transfer model

**Following example needs logging out from the server**.
```bash
rsync -rv --partial --progress user@192.168.0.112:~/dlr/data/2024-11-12-13-14/*.pth ~/dlr/models/pilot.pth
rsync -rv --partial --progress user@192.168.0.112:~/dlr/data/2024-11-12-13-14/*.png ~/dlr/models/  # optional
```
7. Deploy autopilot
```bash
python ~/dlr/scripts/autopilot.py
```

## Requirements
- Design your autopilot model in [convnets.py](scripts/convnets.py). The model is suppose to take in color image with shape of `(176, 208, 3)` and output steering and throttle values with shape of `(1, 2)`. 
- Collect data to train your autopilot.
- Deploy and test the autopilot model.

## Rubric 
- **(100%) The deployed autopilot is expected to finish at least one lap of the track autonomously.** Any human correction/interference will cost 5% of the total score.
- For the final demonstration, set and start the BearCart at the "Start/Finish Line".
- Release the autopilot after the instructor's verbal cue.
- Operators may follow and correct the robot if any unexpected situation (crash, stuck, off-track, etc.) happened. Be familiar with the `PAUSE` button.
- The time cost to finish a lap and the number of human corrections/interferences will be recorded.
- Each team has 5 attempts. Each attempt should not last over 2 minutes.
- The autopilot model will be tested and showcased on the track as shown below.
![race_track](111_raceway.png)
