
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator --num_hidden_layers=5 --latent_dim=100 --lr_decoder=0.0005;

python3 generate_data_realistic.py v3
# cd models
# python baseline.py v3
# cd ..
# python3 generate_classifier_data.py v3

# python3 train_classifier.py --config_file=configs/classifier/classifier.yaml;

# python3 train_finetuner.py --config_file=configs/finetuner/finetuner_nobaseline_low.yaml;
python3 train_finetuner.py --config_file=configs/finetuner/finetuner_nobaseline_large.yaml;

# python train_errorfinder.py --config_file=configs/errorfinder/errorfinder.yaml;

