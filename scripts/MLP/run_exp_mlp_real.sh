python src/run_experiment.py \
	--model 'mlp' \
	--mlp_activation 'tanh' \
	--mlp_hidden_layer_list 100 100 10 \
	--dataset 'metabric-pam50__200' \
	--lr 0.001 \
	--max_steps 5000 \
	--metric_model_selection cross_entropy_loss \
	--tags 'eval'
