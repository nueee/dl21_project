# dl21_project

## directory structure
```
dl21_project
ㄴ checkpoints (should be added manually by $ mkdir checkpoints)
    ㄴ [subfolder named with your current trial name] (should be added manually by $ mkdir checkpoints/{trial_name})
    ...
ㄴ intermediate_results (should be added manually by mkdir intermediate_results)
    ㄴ [subfolder named with your current trial name] (should be added manually by $ mkdir checkpoints/{trial_name})
    ...
ㄴ dataset
    ㄴ cartoons
        ㄴ 1
            ㄴ (image files such as selfie2anime_anime...)
    ㄴ photos
        ㄴ 1
            ㄴ (image files such as selfie2anime_photo...)
    ㄴ cartoons_smoothed
        ㄴ 1
            ㄴ (image files such as selfie2anime_anime_smoothed...)
ㄴ tensorboard (will be made automatically)
    ㄴ [subfolder named with your current trial name]
    ...
main.py (almost same with .ipynb in the google drive)
loss_functions.py
neural_nets.py
prepare_data.py
trainers.py
```
원하는 .py 추가 가능! (train-new only, train-continue only, test only, etc.)
