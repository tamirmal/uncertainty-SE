import torch
import torch.nn as nn

"""
This file provides the code for the uncertainty model (Aleatoric-WF & Aleatoric-AMAP) in the paper

Huajian Fang, Dennis Becker, Stefan Wermter, Timo Gerkmann, "Integrating Uncertainty into Neural Network-based Speech Enhancement",
IEEE/ACM Trans. Audio, Speech, Language Proc., accepted for publication.
"""

EPS = 1e-8

def conv_insn_lrelu(in_channel, out_channel, kernel_size_in, stride_in, padding_in, insn=True, lrelu=True):
    layers = []
    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size_in, stride=stride_in, padding=padding_in))
    if insn:
        layers.append(nn.InstanceNorm2d(out_channel,affine=False))
    if lrelu:
        layers.append(nn.LeakyReLU(0.2))

    layers = nn.Sequential(*layers)

    return layers

def convt_insn_lrelu(in_channel, out_channel, kernel_size_in, stride_in, padding_in, insn=True, lrelu=True):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size_in, stride=stride_in, padding=padding_in))
    if insn:
        layers.append(nn.InstanceNorm2d(out_channel,affine=False))
    if lrelu:
        layers.append(nn.LeakyReLU(0.2))

    layers = nn.Sequential(*layers)

    return layers


class EDNet_uncertainty(nn.Module):
    def get_type(self):
        return self.model_type

    def __init__(self, input_channel=1, model_type='aleatoric_amap'):
        super(EDNet_uncertainty, self).__init__()
        # processing input: Batch x Input_channel x Time x Frequency
        self.model_type = model_type

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt5 = convt_insn_lrelu(256+256, 128, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt4 = convt_insn_lrelu(128+128, 64, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt3 = convt_insn_lrelu(64+64, 32, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt2 = convt_insn_lrelu(32+32, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1 = convt_insn_lrelu(16+16, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)
        self.convt1_logvar = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)

    # Tamir - some notes:
    #   NN (at least here) - uses real numbers, so this network output magnitude
    #    and then we multiply by noisy_complex to get complex, but what is noisy_complex ?
    #    is it X or Phase(X)
    #
    #  Lets assume that noisy_complex is X (i.e, complex, STFT of input speech)
    #   and x is magnitude - torch.abs(noisy_complex)
    #
    #  This makes sense because the comments below say "Wiener/Approximated_MAP Filtering" - i.e, the actual filtering ...
    #
    def forward(self, x, noisy_complex):
        x = torch.unsqueeze(x, 1) # B, 1, T, F
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        convt6 = self.convt6(conv6)
        y = torch.cat((convt6, conv5), 1)

        convt5 = self.convt5(y)
        y = torch.cat((convt5, conv4), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv3), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv2), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv1), 1) # B, C, T, F

        convt1 = self.convt1(y)

        mean = torch.sigmoid(self.convt1_mean(convt1)).squeeze() # B x 1 x T x F -> (B) x T x F
        logvar = self.convt1_logvar(convt1).squeeze() # B x 1 x T x F  -> (B) x T x F

        # Wiener filtering
        WF_stft = mean * noisy_complex

        # Approximated_MAP filtering
        element = (0.5 * mean)**2 + torch.exp(logvar) / (4 * x.squeeze()**2 + EPS)
        approximated_map = 0.5 * mean + torch.sqrt(element + EPS)
        AMAP_stft = approximated_map * noisy_complex

        return WF_stft, AMAP_stft, logvar

class EDNet_uncertainty_amap(nn.Module):
    def get_type(self):
        return self.model_type

    def __init__(self, input_channel=1,  model_type='amap'):
        super(EDNet_uncertainty_amap, self).__init__()
        # processing input: Batch x Input_channel x Time x Frequency
        self.model_type = model_type

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt5 = convt_insn_lrelu(256+256, 128, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt4 = convt_insn_lrelu(128+128, 64, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt3 = convt_insn_lrelu(64+64, 32, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt2 = convt_insn_lrelu(32+32, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1 = convt_insn_lrelu(16+16, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)
        self.convt1_logvar = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)

    # Tamir - some notes:
    #   NN (at least here) - uses real numbers, so this network output magnitude
    #    and then we multiply by noisy_complex to get complex, but what is noisy_complex ?
    #    is it X or Phase(X)
    #
    #  Lets assume that noisy_complex is X (i.e, complex, STFT of input speech)
    #   and x is magnitude - torch.abs(noisy_complex)
    #
    #  This makes sense because the comments below say "Wiener/Approximated_MAP Filtering" - i.e, the actual filtering ...
    #
    def forward(self, x, noisy_complex):
        x = torch.unsqueeze(x, 1) # B, 1, T, F
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        convt6 = self.convt6(conv6)
        y = torch.cat((convt6, conv5), 1)

        convt5 = self.convt5(y)
        y = torch.cat((convt5, conv4), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv3), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv2), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv1), 1) # B, C, T, F

        convt1 = self.convt1(y)

        mean = torch.sigmoid(self.convt1_mean(convt1)).squeeze() # B x 1 x T x F -> (B) x T x F
        logvar = self.convt1_logvar(convt1).squeeze() # B x 1 x T x F  -> (B) x T x F

        # Approximated_MAP filtering
        element = (0.5 * mean)**2 + torch.exp(logvar) / (4 * x.squeeze()**2 + EPS)
        approximated_map = 0.5 * mean + torch.sqrt(element + EPS)
        AMAP_stft = approximated_map * noisy_complex

        return None, AMAP_stft, logvar


class EDNet_uncertainty_wf_logvar(nn.Module):
    def get_type(self):
        return self.model_type

    def __init__(self, input_channel=1, model_type='wf_logvar'):
        super(EDNet_uncertainty_wf_logvar, self).__init__()
        # processing input: Batch x Input_channel x Time x Frequency
        self.model_type = model_type

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt5 = convt_insn_lrelu(256+256, 128, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt4 = convt_insn_lrelu(128+128, 64, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt3 = convt_insn_lrelu(64+64, 32, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt2 = convt_insn_lrelu(32+32, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1 = convt_insn_lrelu(16+16, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)
        self.convt1_logvar = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)

    # Tamir - some notes:
    #   NN (at least here) - uses real numbers, so this network output magnitude
    #    and then we multiply by noisy_complex to get complex, but what is noisy_complex ?
    #    is it X or Phase(X)
    #
    #  Lets assume that noisy_complex is X (i.e, complex, STFT of input speech)
    #   and x is magnitude - torch.abs(noisy_complex)
    #
    #  This makes sense because the comments below say "Wiener/Approximated_MAP Filtering" - i.e, the actual filtering ...
    #
    def forward(self, x, noisy_complex):
        x = torch.unsqueeze(x, 1) # B, 1, T, F
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        convt6 = self.convt6(conv6)
        y = torch.cat((convt6, conv5), 1)

        convt5 = self.convt5(y)
        y = torch.cat((convt5, conv4), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv3), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv2), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv1), 1) # B, C, T, F

        convt1 = self.convt1(y)

        mean = torch.sigmoid(self.convt1_mean(convt1)).squeeze() # B x 1 x T x F -> (B) x T x F
        logvar = self.convt1_logvar(convt1).squeeze() # B x 1 x T x F  -> (B) x T x F

        # Wiener filtering
        WF_stft = mean * noisy_complex

        return mean, None, logvar


class EDNet_uncertainty_baseline_wf(nn.Module):
    def get_type(self):
        return self.model_type

    def __init__(self, input_channel=1, model_type='baseline_wf'):
        super(EDNet_uncertainty_baseline_wf, self).__init__()

        self.model_type = model_type
        # processing input: Batch x Input_channel x Time x Frequency

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt5 = convt_insn_lrelu(256+256, 128, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt4 = convt_insn_lrelu(128+128, 64, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt3 = convt_insn_lrelu(64+64, 32, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt2 = convt_insn_lrelu(32+32, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1 = convt_insn_lrelu(16+16, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)

    # Tamir - some notes:
    #   NN (at least here) - uses real numbers, so this network output magnitude
    #    and then we multiply by noisy_complex to get complex, but what is noisy_complex ?
    #    is it X or Phase(X)
    #
    #  Lets assume that noisy_complex is X (i.e, complex, STFT of input speech)
    #   and x is magnitude - torch.abs(noisy_complex)
    #
    #  This makes sense because the comments below say "Wiener/Approximated_MAP Filtering" - i.e, the actual filtering ...
    #
    def forward(self, x, noisy_complex):
        x = torch.unsqueeze(x, 1) # B, 1, T, F
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        convt6 = self.convt6(conv6)
        y = torch.cat((convt6, conv5), 1)

        convt5 = self.convt5(y)
        y = torch.cat((convt5, conv4), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv3), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv2), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv1), 1) # B, C, T, F

        convt1 = self.convt1(y)

        mean = torch.sigmoid(self.convt1_mean(convt1)).squeeze() # B x 1 x T x F -> (B) x T x F
        WF_stft = mean * noisy_complex

        return WF_stft, None, None


# this network should use the loss in (8)
class EDNet_uncertainty_epistemic_dropout(nn.Module):
    def get_type(self):
        return 'mc-dropout'

    def get_M(self):
        return self.M

    def __init__(self, input_channel=1, M = 8):
        super(EDNet_uncertainty_epistemic_dropout, self).__init__()
        self.M = M

        # processing input: Batch x Input_channel x Time x Frequency

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.dropout4 = nn.Dropout(0.5)
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.dropout5 = nn.Dropout(0.5)
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5,5), stride_in=(1,2), padding_in=(2,2))
        self.dropout6 = nn.Dropout(0.5)

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt5 = convt_insn_lrelu(256+256, 128, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt4 = convt_insn_lrelu(128+128, 64, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt3 = convt_insn_lrelu(64+64, 32, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt2 = convt_insn_lrelu(32+32, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1 = convt_insn_lrelu(16+16, 16, kernel_size_in=(5,5),stride_in=(1,2), padding_in=(2,2))
        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1, padding_in=0, insn=False, lrelu=False)

    def enable_dropout(self, mode=True):
        if mode:
            self.dropout4.train()
            self.dropout5.train()
            self.dropout6.train()
        else:
            self.dropout4.eval()
            self.dropout5.eval()
            self.dropout6.eval()

    def forward(self, x, noisy_complex):
        x = torch.unsqueeze(x, 1) # B, 1, T, F
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4d = self.dropout4(conv4)
        conv5 = self.conv5(conv4d)
        conv5d = self.dropout5(conv5)
        conv6 = self.conv6(conv5d)
        conv6d = self.dropout6(conv6)

        convt6 = self.convt6(conv6d)
        y = torch.cat((convt6, conv5d), 1)

        convt5 = self.convt5(y)
        y = torch.cat((convt5, conv4d), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv3), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv2), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv1), 1) # B, C, T, F

        convt1 = self.convt1(y)

        mean = torch.sigmoid(self.convt1_mean(convt1)).squeeze() # B x 1 x T x F -> (B) x T x F

        # Wiener filtering
        WF_stft = mean * noisy_complex

        return WF_stft, None, None


if __name__ == "__main__":

    input = torch.randn(16,150,257, dtype=torch.float).cuda()
    noisy = torch.randn(16,150,257, dtype=torch.cfloat).cuda()
    model = EDNet_uncertainty().cuda()
    output = model(input, noisy)

    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
