# XAI_MultiSpan_MachineReadingComprehension


# Dependencies
* python 3.7
* PyTorch 1.6.0
* Transformers 2.11.0
* AttrDict

# Model Architecture
![Model](https://user-images.githubusercontent.com/80243108/204685002-ec31815d-ddb6-4336-b0a1-353ea264db7a.jpg)

# Data
* Quoref
* MASHQA

# Train & Test
* python3.7 run_mrc --train_file [train file] --test_file [test_file] --from_init_weight --do_train
* python3.7 run_mrc --test_file [test_file] --do_evaluate
