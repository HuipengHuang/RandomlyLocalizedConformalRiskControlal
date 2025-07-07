from .ae import AutoEncoder
from .vae import VariationalAutoEncoder
from .sparse_vae import SparseVAE
def get_vae(args, input_dim):
    if args.vae == "vae":
        return VariationalAutoEncoder(input_dim=input_dim, latent_dim=args.latent_dim if args.latent_dim else 10)
    elif args.vae == "svae":
        return SparseVAE(input_dim=input_dim, latent_dim=args.latent_dim if args.latent_dim else 10)
    elif args.vae == "ae":
        return AutoEncoder(input_dim=input_dim, latent_dim=args.latent_dim if args.latent_dim else 10)
    else:
        raise NotImplementedError