# create the GP layer called after the neural network
# using **one** GP per feature (as in the SV-DKL paper)
### the outputs of these GPs will be mixed in the softmax likelihood
from _shared_imports import *
from custom_models.mlp.MLP import LitMLP
from custom_models.protogate.ProtoGate import LitProtoGate


def create_model(args, data_module=None):
    """
	Function to create the model. Firstly creates the components (e.g., FeatureExtractor, Decoder) and then assambles them.

	Returns a model instance.
	"""
    pl.seed_everything(args.seed_model_init, workers=True)

    if args.model in ['mlp']:
        model = LitMLP(args)
    elif args.model == 'protogate':
        model = LitProtoGate(args)
    else:
        raise Exception(f"The model ${args.model}$ is not supported")

    return model
