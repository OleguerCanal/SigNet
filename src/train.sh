# python train_finetuner.py --config_file=configs/finetuner/finetuner_random_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_random_low.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low.yaml;

python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_random_large.yaml;
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_random_low.yaml;
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_realistic_large.yaml;
python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_realistic_low.yaml;