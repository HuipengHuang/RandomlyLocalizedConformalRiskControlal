from .thr import THR
from .aps import APS

def get_score(args):
    if args.score == "thr":
        return THR()

    elif args.random is None:
        # APS, RAPS, SAPS all need to choose random version or determined version.
        assert args.random is not None, "Please specify --random."

    elif args.score == "aps":
        return APS((args.random == "True"))

    # If no valid score function is found
    raise RuntimeError("Cannot find a suitable score function.")
