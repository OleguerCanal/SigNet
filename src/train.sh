# cd models
# python baseline.py
# cd ..


python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator --num_hidden_layers=5 --latent_dim=100 --lr_decoder=0.0005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator1 --num_hidden_layers=5 --latent_dim=150 --lr_decoder=0.0005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator2 --num_hidden_layers=8 --latent_dim=150 --lr_decoder=0.0005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator3 --num_hidden_layers=5 --latent_dim=200 --lr_decoder=0.0005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator4 --num_hidden_layers=5 --latent_dim=2000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator4 --num_hidden_layers=8 --latent_dim=2000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator5 --num_hidden_layers=5 --latent_dim=5000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator6 --num_hidden_layers=8 --latent_dim=5000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator7 --num_hidden_layers=5 --latent_dim=10000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator8 --num_hidden_layers=8 --latent_dim=10000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator9 --num_hidden_layers=5 --latent_dim=5000 --lr_encoder=0.0005 --lr_decoder=0.00005;
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator10 --num_hidden_layers=8 --latent_dim=50000 --lr_encoder=0.0005 --lr_decoder=0.00005;

# python train_classifier.py --config_file=configs/classifier/classifier.yaml;

# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_low.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low.yaml;

# python train_errorfinder.py --config_file=configs/errorfinder/errorfinder.yaml;

