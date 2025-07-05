import torch.nn as nn
import torch

class FeatureModel(nn.Module):
    def __init__(self, net, args):
        super(FeatureModel, self).__init__()
        self.net = net
        self.args = args

        # Feature extractor: all layers except final fc and avgpool
        self.feature_extractor = nn.Sequential(*list(net.children())[:-2])

        # Original avgpool and fc
        self.avgpool = net.avgpool
        self.MLP = net.fc

    def get_feature(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        return features.flatten(1)  # Flatten all dimensions except batch

    def feature2logits(self, feature):
        return self.MLP(feature)

    def eval(self):
        self.feature_extractor.eval()
        self.MLP.eval()
        self.avgpool.eval()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = features.flatten(1)  # Critical: flatten before MLP
        return self.MLP(features)

class LogitsModel(nn.Module):
    def __init__(self, net, args):

        super(LogitsModel, self).__init__()
        self.net = net
        self.args = args

    def get_feature(self, x):
        return self.net(x)

    def feature2logits(self, feature):
        return feature

    def eval(self):
        self.net.eval()

    def forward(self, x):
        return self.net(x)

class SoftmaxModel(nn.Module):
    def __init__(self, net, args):
        super(SoftmaxModel, self).__init__()
        self.net = net
        self.args = args

    def get_feature(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=1)

    def feature2logits(self, feature):
        raise NotImplementedError

    def eval(self):
        self.net.eval()

    def forward(self, x):
        return self.net(x)

def get_model(net, args):
    if args.feature == "feature":
        return FeatureModel(net, args)
    elif args.feature == "logits":
        return LogitsModel(net, args)
    elif args.feature == "softmax":
        return SoftmaxModel(net, args)
    else:
        raise NotImplementedError