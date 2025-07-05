from .box_kernel import BoxKernel
from .gaussian_kernel import GaussianKernel
def get_kernel_function(args, holdout_feature=None, holdout_target=None):
    if args.kernel_function == "gaussian":
        kernel_function = GaussianKernel(args, holdout_feature=holdout_feature, holdout_target=holdout_target, h=args.h)
    elif args.kernel_function == "box":
        kernel_function = BoxKernel(args, holdout_feature=holdout_feature, holdout_target=holdout_target, h=args.h)
    else:
        raise NotImplementedError("Not implemented")
    return kernel_function