import functools
import torch
import torch.nn as nn


def cal_conv_feature(L_in, kernel_size, stride, padding):
    return (L_in - kernel_size + 2*padding) // stride + 1


def cal_dconv_feature(L_in, kernel_size, stride, padding, dilation, output_padding):
    return (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


# construct autoencoder
class encoder(nn.Module):
    def __init__(self, data_len, inplanes, planes, kernel_size, stride=1, padding=0):
        super(encoder, self).__init__()
        self.feature_len = cal_conv_feature(data_len, kernel_size, stride, padding)
        self.conv1 = nn.Conv1d(inplanes, planes, 
                               kernel_size=kernel_size,
                               stride=stride, bias=False, padding=padding)
        self.bn1 = nn.BatchNorm1d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# construct autoencoder
class decoder(nn.Module):
    def __init__(self, data_len, data_len_match, inplanes, planes, kernel_size, stride=1, padding=0,
                dilation=1, output_padding=0):
        super(decoder, self).__init__()
        self.feature_len = cal_dconv_feature(data_len, kernel_size, stride, padding, dilation, output_padding)
        if data_len_match > self.feature_len:
            output_padding = data_len_match - self.feature_len
        else:
            output_padding = 0
        self.feature_len = cal_dconv_feature(data_len, kernel_size, stride, padding, dilation, output_padding)

        self.conv1 = nn.ConvTranspose1d(inplanes, planes, 
                                        kernel_size=kernel_size, 
                                        stride=stride, bias=False, 
                                        padding=padding,
                                        dilation=dilation,
                                        output_padding=output_padding)
        self.bn1 = nn.BatchNorm1d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# used for similHuber-Huber-simpleDiscr
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        kernel_size = 5
        filters_num = 16
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=filters_num, kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=filters_num, out_channels=filters_num*2, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(filters_num*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=filters_num*2, out_channels=filters_num*4, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(filters_num*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=filters_num*4, out_channels=filters_num*8, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(filters_num*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last = nn.Conv1d(in_channels=filters_num*8, out_channels=1, kernel_size=kernel_size, stride=1, padding='same')

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.last(x)
        return x


# used for vector-GAN-PixelD
class PixelDiscriminator(nn.Module):
    def __init__(self):
        super(PixelDiscriminator, self).__init__()
        ndf = 64
        norm_layer = nn.BatchNorm1d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm1d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.net = [
            nn.Conv1d(2, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# used for vector-GAN-PatchG
class PatchGAN_Discriminator(nn.Module):
    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        super(PatchGAN_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm1d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# used for VECTOR-GAN, SimilHuber
class VECTOR_GAN_Discriminator(nn.Module):
    def __init__(self):
        super(VECTOR_GAN_Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1, kernel_size=8, stride=1, padding=0)

    def forward(self, input):
        x = self.conv1(input)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        return x


# used for VECTOR-GAN
class CAE_8_Generator(nn.Module):
    def __init__(self, data_len, is_skip=True):
        super(CAE_8_Generator, self).__init__()
        kernel_size = 8
        self.encoder1 = encoder(data_len=data_len, inplanes=1, planes=64, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en1_num_feature = self.encoder1.feature_len
        self.encoder2 = encoder(data_len=self.en1_num_feature, inplanes=64, planes=128, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en2_num_feature = self.encoder2.feature_len
        self.encoder3 = encoder(data_len=self.en2_num_feature, inplanes=128, planes=256, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en3_num_feature = self.encoder3.feature_len
        self.encoder4 = encoder(data_len=self.en3_num_feature, inplanes=256, planes=512, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en4_num_feature = self.encoder4.feature_len
        self.encoder5 = encoder(data_len=self.en4_num_feature, inplanes=512, planes=1024, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en5_num_feature = self.encoder5.feature_len
        self.encoder6 = encoder(data_len=self.en5_num_feature, inplanes=1024, planes=2048, 
                                kernel_size=kernel_size, stride=2, padding=0)
        self.en6_num_feature = self.encoder6.feature_len
        self.encoder7 = encoder(data_len=self.en6_num_feature, inplanes=2048, planes=2048, 
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en7_num_feature = self.encoder7.feature_len
        self.encoder8 = encoder(data_len=self.en7_num_feature, inplanes=2048, planes=2048,
                                kernel_size=kernel_size, stride=1, padding=0)
        self.en8_num_feature = self.encoder8.feature_len

        self.decoder1 = decoder(data_len=self.en8_num_feature, data_len_match=self.en7_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder2 = decoder(data_len=self.en7_num_feature, data_len_match=self.en6_num_feature, 
                                inplanes=2048, planes=2048, kernel_size=kernel_size, stride=1, padding=0)
        self.decoder3 = decoder(data_len=self.en6_num_feature, data_len_match=self.en5_num_feature, 
                                inplanes=2048, planes=1024, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder4 = decoder(data_len=self.en5_num_feature, data_len_match=self.en4_num_feature, 
                                inplanes=1024, planes=512, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder5 = decoder(data_len=self.en4_num_feature, data_len_match=self.en3_num_feature, 
                                inplanes=512, planes=256, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder6 = decoder(data_len=self.en3_num_feature, data_len_match=self.en2_num_feature,
                                inplanes=256, planes=128, kernel_size=kernel_size, stride=2, padding=0)
        self.decoder7 = decoder(data_len=self.en2_num_feature, data_len_match=self.en1_num_feature,
                                inplanes=128, planes=64, kernel_size=kernel_size, stride=2, padding=0)

        kernel_size = kernel_size
        stride = 1
        output_padding = 0
        padding = 0
        dilation = 1

        last_feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding,
                                             dilation, output_padding)
        if data_len > last_feature_len:
            padding = data_len - last_feature_len
        else:
            padding = 0
        self.feature_len = cal_dconv_feature(self.en1_num_feature, kernel_size, stride, padding,
                                             dilation, output_padding)
        self.decoder8 = nn.ConvTranspose1d(64, 1, 
                                           kernel_size=kernel_size, stride=stride,
                                           bias=False,
                                           padding=padding,
                                           dilation=dilation,
                                           output_padding=output_padding)
        self.sigmoid = nn.Sigmoid()  
        self.is_skip = is_skip

    def forward(self, x):
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)
        en6 = self.encoder6(en5)
        en7 = self.encoder7(en6)
        encoding = self.encoder8(en7)

        de1 = self.decoder1(encoding)
        if self.is_skip:
            de2 = self.decoder2(de1+en7)
            de3 = self.decoder3(de2+en6)
            de4 = self.decoder4(de3+en5)
            de5 = self.decoder5(de4+en4)
            de6 = self.decoder6(de5+en3)
            de7 = self.decoder7(de6+en2)
        else:
            de2 = self.decoder2(de1)
            de3 = self.decoder3(de2)
            de4 = self.decoder4(de3)
            de5 = self.decoder5(de4)
            de6 = self.decoder6(de5)
            de7 = self.decoder7(de6)
        de8 = self.decoder8(de7)
        out = self.sigmoid(de8)
        return out


# used for similHuber
class SimilHuber_Generator(nn.Module):
    def __init__(self, data_len, is_skip=True):
        super(SimilHuber_Generator, self).__init__()
        filters_num = 32
        self.activation = nn.LeakyReLU(inplace=True)  # activation layer (same for all)

        # first downsampling layers
        self.conv1_len = cal_conv_feature(data_len, 9, 1, 0)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters_num, kernel_size=9,
                               bias=False, stride=1)

        self.conv2_len = cal_conv_feature(self.conv1_len, 9, 2, 0)
        self.conv2 = nn.Conv1d(in_channels=filters_num, out_channels=filters_num, kernel_size=9,
                               bias=False, stride=2)

        # second downsampling layers
        self.conv3_len = cal_conv_feature(self.conv2_len, 7, 1, 0)
        self.conv3 = nn.Conv1d(in_channels=filters_num, out_channels=filters_num*2, kernel_size=7,
                               bias=False, stride=1)
        self.conv4_len = cal_conv_feature(self.conv3_len, 7, 2, 0)
        self.conv4 = nn.Conv1d(in_channels=filters_num*2, out_channels=filters_num*2, kernel_size=7,
                               bias=False, stride=2)

        # third downsampling layers
        self.conv5_len = cal_conv_feature(self.conv4_len, 3, 1, 0)
        self.conv5 = nn.Conv1d(in_channels=filters_num*2, out_channels=filters_num*4, kernel_size=3,
                               bias=False, stride=1)

        # first upsampling layer
        self.deconv1_len = cal_dconv_feature(self.conv5_len, 3, 1, 0, 1, 0)
        output_p1 = 0
        if self.conv4_len > self.deconv1_len:
            output_p1 = self.conv4_len - self.deconv1_len
            self.deconv1_len = cal_dconv_feature(self.conv5_len, 3, 1, 0, 1, output_p1)
        self.deconv1 = nn.ConvTranspose1d(filters_num*4, filters_num*4, kernel_size=3, 
                                          stride=1, bias=False, padding=0, dilation=1, output_padding=output_p1)
        
        self.deconv2_len = self.deconv1_len
        self.deconv2 = nn.Conv1d(in_channels=filters_num*4+filters_num*2, out_channels=filters_num*2, kernel_size=3,
                                 bias=False, stride=1, padding='same')

        # second upsampling layer
        self.deconv3_len = cal_dconv_feature(self.deconv2_len, 7, 2, 0, 2, 0)
        output_p3 = 0
        if self.conv2_len > self.deconv3_len:
            output_p3 = self.conv2_len - self.deconv3_len
            self.deconv3_len = cal_dconv_feature(self.deconv2_len, 7, 2, 0, 2, output_p3)
  
        self.deconv3 = nn.ConvTranspose1d(filters_num*2, filters_num*2, kernel_size=7, 
                                          stride=2, bias=False, padding=0, dilation=2, output_padding=output_p3)
        
        self.deconv4_len = self.deconv3_len
        self.deconv4 = nn.Conv1d(in_channels=filters_num*2+filters_num, out_channels=filters_num, kernel_size=7,
                                 bias=False, stride=1, padding='same')

        # third upsampling layer
        self.deconv5_len = cal_dconv_feature(self.deconv4_len, 9, 2, 0, 2, 0)
        output_p5 = 0
        if data_len > self.deconv5_len:
            output_p5 = data_len - self.deconv5_len
            self.deconv5_len = cal_dconv_feature(self.deconv2_len, 9, 2, 0, 2, output_p5)
  
        self.deconv5 = nn.ConvTranspose1d(filters_num, filters_num, kernel_size=9, 
                                          stride=2, bias=False, padding=0, dilation=2, output_padding=output_p5)
        
        self.third_last = nn.Conv1d(in_channels=filters_num+1, out_channels=1, kernel_size=9, bias=False,
                                    stride=1, padding='same')

        self.second_last = nn.Linear(data_len, 3*data_len)
        self.last = nn.Linear(3*data_len, data_len)

        self.batch = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        r1 = self.activation(c2)

        c3 = self.conv3(r1)
        c4 = self.conv4(c3)
        r2 = self.activation(c4)

        c5 = self.conv5(r2)

        d1 = self.deconv1(c5)
        conc1 = torch.cat((d1, r2), dim=1)
        d2 = self.deconv2(conc1)
        r3 = self.activation(d2)

        d3 = self.deconv3(r3)
        conc2 = torch.cat((d3, c2), dim=1)
        d4 = self.deconv4(conc2)
        r5 = self.activation(d4)

        d5 = self.deconv5(r5)
        conc3 = torch.cat((d5, x), dim=1)

        out = self.third_last(conc3)

        # out = self.batch(out)
        out = self.activation(out)
        out = self.second_last(out)
        out = self.activation(out)

        out = self.drop(out)

        out = self.last(out)
        out = nn.Sigmoid()(out)
        # out = nn.ReLU()(out)

        return out
