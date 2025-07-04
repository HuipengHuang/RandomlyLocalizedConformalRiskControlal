import torch.nn as nn
class Model(nn.Module):
    def __init__(self, net, args):
        super(Model, self).__init__()
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