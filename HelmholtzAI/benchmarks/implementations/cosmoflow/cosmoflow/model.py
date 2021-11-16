import mxnet as mx
import mxnet.gluon.nn as nn

import numpy as np

import utils

from typing import Literal, Optional


class DefaultInitializer(mx.init.Xavier):
    def __init__(self, layout=Literal["NCDHW", "NDHWC"]):
        super().__init__()
        self.layout=layout

    def _init_weight(self, name, arr):
        if len(arr.shape) != 5:
            return super()._init_weight(name, arr)
        self._init_conv3d_weight(arr)
        

    def _init_conv3d_weight(self, arr):
        if self.layout == "NDHWC":
            fan_in, fan_out = arr.shape[-1], arr.shape[0]
            prod = np.prod(arr.shape[1:-1])
        else:
            fan_in, fan_out = arr.shape[1], arr.shape[0]
            prod = np.prod(arr.shape[2:])
        
        scale = np.sqrt(6.0 / ((fan_in + fan_out) * prod))
        mx.nd.random.uniform(-scale, scale, shape=arr.shape, out=arr)
        


class PaddedConvolution(nn.HybridBlock):
    def __init__(self, conv_channels: int, kernel_size: int,
                 layout: Optional[Literal["NCDHW", "NDHWC"]] = None):
        super().__init__()
        if layout is None:
            layout = "NCDHW" if kernel_size & 1 == 0 else "NDHWC"
        
        with self.name_scope():
            in_layer_padding = (kernel_size // 2) if (kernel_size & 1) else 0
            self.kernel_size = kernel_size
            self.convolution = nn.Conv3D(channels=conv_channels,
                                         kernel_size=kernel_size,
                                         padding=in_layer_padding,
                                         layout=layout)
            self.activation = nn.LeakyReLU(alpha=0.3)
            #self.activation = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.kernel_size & 1 == 0:
            x = F.pad(x, mode="constant", constant_value=0,
                      pad_width=(0, 0, 0, 0, 0, self.kernel_size // 2,
                                 0, self.kernel_size // 2,
                                 0, self.kernel_size // 2))
        return self.activation(self.convolution(x))


class Scale1p2(nn.HybridBlock):
    def __init__(self):
        super().__init__()

    def hybrid_forward(self, F, x):
        return x * 1.2


def add_dense_block(network_container: nn.HybridSequential, 
                    dense_unit: int, lrelu_alpha: float, 
                    dropout_rate: Optional[float] = None):
    network_container.add(nn.Dense(dense_unit),
                          nn.LeakyReLU(alpha=lrelu_alpha))
    if dropout_rate is not None:
        network_container.add(nn.Dropout(dropout_rate))


def build_cosmoflow_model(n_conv_layers: int = 5,
                          conv_kernel: int = 2,
                          dropout_rate: float = 0.5,
                          layout: Literal["NCDHW", "NDHWC"] = "NDHWC",
                          use_wd: bool = False) -> nn.HybridSequential:
    cosmoflow = nn.HybridSequential()

    for i in range(n_conv_layers):
        cosmoflow.add(PaddedConvolution(conv_channels=32 * (1 << i),
                                        kernel_size=conv_kernel,
                                        layout=layout),
                      nn.MaxPool3D(pool_size=2, layout=layout))

    dropout_rate = dropout_rate if not use_wd else None
    add_dense_block(cosmoflow, 128, 0.3, dropout_rate)
    add_dense_block(cosmoflow, 64, 0.3, dropout_rate)

    cosmoflow.add(nn.Dense(4),
                  nn.Activation("tanh"),
                  Scale1p2())

    return cosmoflow
                                    
class CosmoflowWithLoss(nn.HybridBlock):
    def __init__(self, 
                 n_conv_layers: int = 5,
                 conv_kernel: int = 2,
                 dropout_rate: float = 0.5,
                 layout: Literal["NCDHW", "NDHWC"] = "NDHWC",
                 use_wd: bool = False):
        super().__init__()
        self.layout = layout
        self.model = build_cosmoflow_model(n_conv_layers, conv_kernel,
                                           dropout_rate, layout, use_wd)
        self.loss_fn = mx.gluon.loss.L2Loss()

    def hybrid_forward(self, F, x, y_true):
        y_pred = self.model(x)#.astype(np.float32)
        return self.loss_fn(y_pred, y_true)

    def init(self, ctx, batch_size: int, use_wd: bool, 
             dist_desc: utils.DistributedEnvDesc,
             checkpoint: Optional[str] = None):

        if checkpoint is not None and checkpoint != "":
            self.load_parameters(checkpoint, ctx=ctx)
        else:
            self.model.initialize(init=DefaultInitializer(layout=self.layout),
                                  ctx=ctx)
        if use_wd:
            self.model.collect_params(select=".*conv.*_weight|.*_bias").setattr("wd_mult", 0.0)

        input_shape = (batch_size, 4, 128, 128, 128) if self.layout == "NCDHW" \
                else (batch_size, 128, 128, 128, 4)
        random_batch = (mx.nd.random.uniform(shape=input_shape, 
                                            ctx=ctx,
                                            dtype="float32"),
                        mx.nd.random.uniform(shape=(batch_size, 4),
                                            ctx=ctx,
                                            dtype="float32"))
        
        if dist_desc.master:
            self.model.summary(random_batch[0])
            

        #self.hybridize(static_alloc=True, static_shape=True)
        self.model.hybridize(static_alloc=True, static_shape=True)

        # warmup
        _ = self(*random_batch)


