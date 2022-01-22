
python train_generator.py --config_file=configs/generator/generator_v2.yaml

python generate_data_realistic.py 'v2'
python generate_data_perturbed.py 'v2'
python generate_classifier_data.py 'v2'

cd models
python baseline.py 'v2'
cd ..

python train_classifier.py --config_file=configs/classifier/classifier_v2.yaml;

python train_finetuner.py --config_file=configs/finetuner/finetuner_low_v2.yaml;
python train_finetuner.py --config_file=configs/finetuner/finetuner_large_v2.yaml;

python train_errorfinder.py --config_file=configs/errorfinder/errorfinder_v2.yaml;

