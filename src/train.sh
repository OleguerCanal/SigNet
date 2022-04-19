# cd models
# python baseline.py
# cd ..


# python3 train_finetuner.py --config_file=configs/finetuner/finetuner_large.yaml --model_id="finetuner_augmented_large" --source="augmented_real_large"
# python3 train_finetuner.py --config_file=configs/finetuner/finetuner_nobaseline_low.yaml --model_id="finetuner_augmented_low" --source="augmented_real_low"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_1" --latent_dim="10"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_2" --latent_dim="20"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_3" --latent_dim="30"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_4" --latent_dim="40"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_5" --latent_dim="50"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_6" --latent_dim="60"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_7" --latent_dim="70"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_8" --latent_dim="80"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_9" --latent_dim="90"
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_10" --latent_dim="100"

python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_11" --latent_dim="10" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_12" --latent_dim="20" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_13" --latent_dim="30" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_14" --latent_dim="40" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_15" --latent_dim="50" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_16" --latent_dim="60" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_17" --latent_dim="70" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_18" --latent_dim="80" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_19" --latent_dim="90" --lagrange_param=0.0001
python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="cancer_type_oversampler_20" --latent_dim="100" --lagrange_param=0.0001

# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="generator_3" --lagrange_param=0.001 --latent_dim=40 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_4" --lagrange_param=0.001 --latent_dim=40 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_5" --lagrange_param=0.0001 --latent_dim=40 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_6" --lagrange_param=0.0001 --latent_dim=40 --num_hidden_layers=4  --lr_encoder=0.005 --lr_decoder=0.0005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_7" --lagrange_param=0.01 --latent_dim=20 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_8" --lagrange_param=0.01 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_9" --lagrange_param=0.001 --latent_dim=20 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_10" --lagrange_param=0.001 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_11" --lagrange_param=0.0001 --latent_dim=20 --num_hidden_layers=2  --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_12" --lagrange_param=0.0001 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_13" --lagrange_param=0.01 --latent_dim=60 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_14" --lagrange_param=0.01 --latent_dim=60 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id="generator_300" --lagrange_param=0.00001 --latent_dim=50 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_301" --lagrange_param=0.0001 --latent_dim=100 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_17" --lagrange_param=0.0001 --latent_dim=60 --num_hidden_layers=2  --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_18" --lagrange_param=0.0001 --latent_dim=60 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_19" --lagrange_param=0.01 --latent_dim=80 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_20" --lagrange_param=0.01 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_21" --lagrange_param=0.001 --latent_dim=80 --num_hidden_layers=2 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_22" --lagrange_param=0.001 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_23" --lagrange_param=0.0001 --latent_dim=80 --num_hidden_layers=2  --lr_encoder=0.005 --lr_decoder=0.0005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_24" --lagrange_param=0.0001 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.005 --lr_decoder=0.0005




# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_101" --lagrange_param=0.01 --latent_dim=40 --num_hidden_layers=2  --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_102" --lagrange_param=0.01 --latent_dim=40 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_103" --lagrange_param=0.001 --latent_dim=40 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_104" --lagrange_param=0.001 --latent_dim=40 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_105" --lagrange_param=0.0001 --latent_dim=40 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_106" --lagrange_param=0.0001 --latent_dim=40 --num_hidden_layers=4  --lr_encoder=0.05 --lr_decoder=0.005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_107" --lagrange_param=0.01 --latent_dim=20 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_108" --lagrange_param=0.01 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_109" --lagrange_param=0.001 --latent_dim=20 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_110" --lagrange_param=0.001 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_111" --lagrange_param=0.0001 --latent_dim=20 --num_hidden_layers=2  --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_112" --lagrange_param=0.0001 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_113" --lagrange_param=0.01 --latent_dim=60 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_114" --lagrange_param=0.01 --latent_dim=60 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_115" --lagrange_param=0.001 --latent_dim=60 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_116" --lagrange_param=0.001 --latent_dim=60 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_117" --lagrange_param=0.0001 --latent_dim=60 --num_hidden_layers=2  --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_118" --lagrange_param=0.0001 --latent_dim=60 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_119" --lagrange_param=0.01 --latent_dim=80 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_120" --lagrange_param=0.01 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_121" --lagrange_param=0.001 --latent_dim=80 --num_hidden_layers=2 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_122" --lagrange_param=0.001 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_123" --lagrange_param=0.0001 --latent_dim=80 --num_hidden_layers=2  --lr_encoder=0.05 --lr_decoder=0.005
# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_124" --lagrange_param=0.0001 --latent_dim=80 --num_hidden_layers=4 --lr_encoder=0.05 --lr_decoder=0.005

# python3 train_generator.py --config_file=configs/generator/generator.yaml --lagrange_param=0.001 --model_id="generator_125" --lagrange_param=0.001 --latent_dim=20 --num_hidden_layers=4 --lr_encoder=0.001 --lr_decoder=0.001


# python3 train_generator.py --config_file=configs/generator/generator.yaml --model_id=generator10 --num_hidden_layers=8 --latent_dim=50000 --lr_encoder=0.0005 --lr_decoder=0.00005;

# python train_classifier.py --config_file=configs/classifier/classifier.yaml;

# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_perturbed_low.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_large.yaml;
# python train_finetuner.py --config_file=configs/finetuner/finetuner_realistic_low.yaml;

# python train_errorfinder.py --config_file=configs/errorfinder/errorfinder.yaml;

