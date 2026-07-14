import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerEncoder


class CrossmodalNet(nn.Module):
    def __init__(self, inchannels, args) -> None:
        super(CrossmodalNet, self).__init__()

        self.modalities = args.modalities
        n_modals = len(args.modalities)

        layers = nn.ModuleDict()

        for j in self.modalities:
            for k in self.modalities:
                if j == k: continue
                layers_name = j + k
                layers[layers_name] = TransformerEncoder(inchannels, num_heads=args.crossmodal_nheads,
                                                         layers=args.num_crossmodal)
            # layers[f'mem_{j}'] = MambaEncoder(inchannels * (n_modals - 1), inchannels * (n_modals - 1),
            #                                   num_layers=args.num_self_att)
            layers[f'mem_{j}'] = TransformerEncoder(inchannels * (n_modals - 1), num_heads=args.self_att_nheads,
                                                    layers=args.num_self_att)

        # layers['composite_at'] = MambaEncoder(inchannels * (n_modals - 1), inchannels * (n_modals - 1),
        #                                       num_layers=args.num_self_att)
        # layers['composite_av'] = MambaEncoder(inchannels * (n_modals - 1), inchannels * (n_modals - 1),
        #                                       num_layers=args.num_self_att)
        # layers['composite_tv'] = MambaEncoder(inchannels * (n_modals - 1), inchannels * (n_modals - 1),
        #                                       num_layers=args.num_self_att)
        layers['composite_at'] = TransformerEncoder(inchannels * (n_modals - 1), num_heads=args.self_att_nheads,
                                                    layers=args.num_self_att)
        layers['composite_av'] = TransformerEncoder(inchannels * (n_modals - 1), num_heads=args.self_att_nheads,
                                                    layers=args.num_self_att)
        layers['composite_tv'] = TransformerEncoder(inchannels * (n_modals - 1), num_heads=args.self_att_nheads,
                                                    layers=args.num_self_att)

        self.layers = layers

    def forward(self, x_s):

        assert len(x_s) == len(self.modalities), f'{len(x_s)} diff {self.modalities}'

        for j in range(len(x_s)):
            x_s[j] = x_s[j].permute(1, 0, 2)

        all_features = []
        out_dict = {}
        for j, x_j in zip(self.modalities, x_s):
            temp = []
            for k, x_k in zip(self.modalities, x_s):
                if j == k: continue
                layer_name = j + k
                out_dict[layer_name] = self.layers[layer_name](x_j, x_k, x_k)
                temp.append(out_dict[layer_name])
            temp = torch.cat(temp, dim=2)
            out_dict[f'mem_{j}'] = self.layers[f'mem_{j}'](temp)
            all_features.append(out_dict[f'mem_{j}'])

        all_features.append(self.layers['composite_at'](torch.cat([out_dict['at'], out_dict['ta']], dim=2)))
        all_features.append(self.layers['composite_av'](torch.cat([out_dict['av'], out_dict['va']], dim=2)))
        all_features.append(self.layers['composite_tv'](torch.cat([out_dict['tv'], out_dict['vt']], dim=2)))
        all_features = torch.cat(all_features, dim=2)

        return all_features
