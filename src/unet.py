### Inspired by: https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture
import torch
import torch.nn as nn


class convolution_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.convolution_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.batch_norm_1 = nn.BatchNorm2d(output_channels)

        self.convolution_2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.convolution_1(inputs)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.convolution_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.convolution = convolution_block(input_channels, output_channels)
        self.pooling = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.convolution(inputs)
        p = self.pooling(x)

        return x, p


class decoder_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.up_convolution = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2, padding=0
        )
        self.convolution = convolution_block(
            output_channels + output_channels, output_channels
        )

    def forward(self, inputs, skip):
        x = self.up_convolution(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.convolution(x)
        return x


class build_unet(nn.Module):
    @staticmethod
    def generate_encoder_blocks(layers: int, input_channels: int, feature_channels: int):
        yield encoder_block(input_channels, feature_channels)
        for i in range(1, layers):
            yield encoder_block(
                feature_channels * 2 ** (i - 1), feature_channels * 2**i
            )

    @staticmethod
    def generate_decoder_blocks(layers: int, feature_channels: int):
        for i in range(layers, 0, -1):
            yield decoder_block(
                feature_channels * 2**i, feature_channels * 2 ** (i - 1)
            )

    def __init__(
        self,
        layers: int = 4,
        input_channels: int = 3,
        feature_channels: int = 64,
        output_channels: int = 1,
    ):
        if not isinstance(layers, int):
            raise TypeError("layers parameter has to be of type int")
        if layers < 1:
            raise ValueError("layers parameter has to be at least 1")
        if not isinstance(input_channels, int):
            raise TypeError("input_channels parameter has to be of type int")
        if input_channels < 1:
            raise ValueError("input_channels parameter has to be at least 1")
        if not isinstance(feature_channels, int):
            raise TypeError("feature_channels parameter has to be of type int")
        if feature_channels < 1:
            raise ValueError("feature_channels parameter has to be at least 1")
        if not isinstance(output_channels, int):
            raise TypeError("output_channels parameter has to be of type int")
        if output_channels < 1:
            raise ValueError("output_channels parameter has to be at least 1")

        super().__init__()

        # Encoder Blocks
        self.encoder_blocks = list(
            self.generate_encoder_blocks(layers, input_channels, feature_channels)
        )

        # Bottleneck
        self.bottleneck = convolution_block(
            feature_channels * 2**(layers-1), feature_channels * 2 ** layers
        )

        # Decoder Blocks
        self.decoder_blocks = list(
            self.generate_decoder_blocks(layers, feature_channels)
        )

        # Classifier
        self.classifier = nn.Conv2d(
            feature_channels, output_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        x = inputs
        skips = []

        # Encoder
        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip)

        # Classification
        y = self.classifier(x)

        return y


if __name__ == "__main__":
    inputs = torch.randn((2, 3, 512, 512))

    c = convolution_block(3, 64)
    x = c(inputs)
    print(x.shape)  # torch.Size([2, 64, 512, 512])

    e = encoder_block(3, 64)
    x, p = e(inputs)
    print(
        x.shape, p.shape
    )  # torch.Size([2, 64, 512, 512]) torch.Size([2, 64, 256, 256])

    pool = torch.randn((2, 64, 256, 256))
    skip = torch.randn((2, 32, 512, 512))
    d = decoder_block(64, 32)
    x = d(pool, skip)
    print(x.shape)  # torch.Size([2, 32, 512, 512])

    model = build_unet()
    outputs = model(inputs)
    print(outputs.shape)  # torch.Size([2, 1, 512, 512])
