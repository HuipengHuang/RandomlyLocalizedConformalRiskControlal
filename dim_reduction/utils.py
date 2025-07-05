from .MLP import DiversifyingMLP
from .PCA.utils import get_pca
from .VAE.utils import get_vae

def get_dimension_reduction_tool(args, holdout_feature, holdout_target):
    if args.pca:
        PCA = get_pca(args, holdout_feature)
        return PCA

    if args.vae:
        VAE = get_vae(args, input_dim=holdout_feature.shape[1]).to("cuda")
        VAE.train()
        VAE.fit(holdout_feature)
        VAE.eval()
        return VAE

    if args.mlp == "True":
        MLP = DiversifyingMLP(input_dim=holdout_feature.shape[1],
                                   output_dim=args.output_dim if args.output_dim else 10).to("cuda")
        MLP.fit(holdout_feature, holdout_target)
        MLP.eval()
        return MLP