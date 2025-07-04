import torch.nn as nn
class Model(nn.Module):
    def __init__(self, net, args):
        super(Model, self).__init__()
        self.net = net
        self.args = args
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
        self.MLP = net.fc

    def get_feature(self, x):
        return self.feature_extractor(x)

    def feature2logits(self, feature):
        return self.MLP(feature)
    def eval(self):
        self.feature_extractor.eval()
        self.MLP.eval()