# cd models
# python baseline.py
# cd ..


# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator --num_hidden_layers=5 --latent_dim=100 --lr_decoder=0.0005;

python3 generate_data_realistic.py 
python3 models/baseline.py 
python3 generate_classifier_data.py

python3 train_classifier.py --config_file=configs/classifier/classifier.yaml;

python3 train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low.yaml;

# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_low.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low.yaml;

# python train_errorfinder.py --config_file=configs/errorfinder/errorfinder.yaml;

