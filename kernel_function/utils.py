from .gaussian_kernel import GaussianKernel
def get_kernel_function(args):
    if args.kernel_function == "gaussian":
        kernel_function = GaussianKernel(h=args.h)
    else:
        raise NotImplementedError("Not implemented")
    return kernel_function