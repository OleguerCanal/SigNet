python train_finetuner.py --config_file=configs/finetuner/finetuner_random_large
python train_finetuner.py --config_file=configs/finetuner/finetuner_random_low
python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_large
python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low

python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_random_large
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_random_low
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_realistic_large
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_realistic_low