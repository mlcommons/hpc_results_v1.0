# copied source from https://github.com/tensorflow/addons/pull/777

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layer-wise Adaptive Rate Scaling optimizer for large-batch training."""

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops

from typeguard import typechecked
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
class MomentumLARS(tf.keras.optimizers.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
      - Gradual learning rate warm-up
      - Linear learning rate scaling
      - Poly rule learning rate decay

    Note, LARS scaling is currently only enabled for dense tensors. Sparse
    tensors use the default momentum optimizer.
    """

    @typechecked
    def __init__(
        self,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        # The LARS coefficient is a hyperparameter
        eeta: float = 0.001,
        epsilon: float = 0.0,
        name: str = "MomentumLARS",
        # Enable skipping variables from LARS scaling.
        skip_list: Optional[str] = (),
        use_nesterov: bool = False,
        clip: bool = False,
        **kwargs
    ):
        """Construct a new LARS Optimizer.

      Args:
          learning_rate: A `Tensor` or floating point value. The base learning rate.
          momentum: A floating point value. Momentum hyperparameter.
          weight_decay: A floating point value. Weight decay hyperparameter. Adding an
             L2 regularizer to a Keras variable is not equivalent to how the LARS paper
             handles weight decay. In the LARS paper, when computing the "trust"
             coefficient, the magnitude of the gradient and the magnitude weights
             are added together. But if an L2 regularizer is added to a Keras variable,
             the gradient and weights are first added together, and then the magnitude is taken
          eeta: LARS coefficient as used in the paper. Default set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the highest
            scaling factor in LARS.
          epsilon: Optional epsilon parameter to be set in models that have very
          small gradients. Default set to 0.0.
          name: Optional name prefix for variables and ops created by LARS Optimizer.
          skip_list: List of strings to enable skipping variables from LARS scaling.
            If any of the strings in skip_list is a subset of var.name, variable
            'var' is skipped from LARS scaling. For a typical classification model
             with batch normalization, the skip_list is ['batch_normalization',
             'bias']
          use_nesterov: when set to True, nesterov momentum will be enabled
          clip: when set to True, learning rate clipping will be enabled.

      Raises:
          ValueError: If a hyperparameter is set to a non-sensical value.
      """
        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay should be positive: %s" % weight_decay)
        super(MomentumLARS, self).__init__(name=name, **kwargs)

        self._skip_list = skip_list
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("eeta", eeta)
        self._set_hyper("epsilon", epsilon)
        self.use_nesterov = use_nesterov
        self.use_clipping = clip

    def get_config(self):
        config = {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "eeta": self._serialize_hyperparameter("eeta"),
            "epsilon": self._serialize_hyperparameter("epsilon"),
            "use_nesterov": self.use_nesterov,
            "clip": self.use_clipping,
        }
        base_config = super(MomentumLARS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    def compute_lr(self, grad, var):
        if self._skip_list is None or not any(v in var.name for v in self._skip_list):
            var_dtype = var.dtype.base_dtype
            w_norm = tf.linalg.norm(var, ord=2)
            g_norm = tf.linalg.norm(grad, ord=2)
            eeta = self._get_hyper("eeta", var_dtype)
            weight_decay = self._get_hyper("weight_decay", var_dtype)
            epsilon = self._get_hyper("epsilon", var_dtype)
            trust_ratio = array_ops.where_v2(
                math_ops.greater(w_norm, 0),
                array_ops.where_v2(
                    math_ops.greater(g_norm, 0),
                    (eeta * w_norm / (g_norm + weight_decay * w_norm + epsilon)),
                    math_ops.cast(1.0, var_dtype),
                ),
                math_ops.cast(1.0, var_dtype),
            )
            lr = self._get_hyper("learning_rate", var_dtype)
            scaled_lr = lr * trust_ratio

            # clip learning rate for LARC.
            if self.use_clipping:
                scaled_lr = min(
                    scaled_lr, lr
                )

            # Add the weight regularization gradient
            grad = grad + weight_decay * var
        return scaled_lr, grad

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, "momentum")
        momentum = self._get_hyper("momentum", var_dtype)
        use_nesterov = self.use_nesterov
        return training_ops.resource_apply_momentum(
            var.handle,
            mom.handle,
            math_ops.cast(1.0, var_dtype),
            grad * scaled_lr,
            momentum,
            use_locking=False,
            use_nesterov=use_nesterov,
        )

    def _resource_apply_sparse(self, grad, var, indices):
        mom = self.get_slot(var, "momentum")
        use_nesterov = self.use_nesterov
        return training_ops.resource_sparse_apply_momentum(
            var.handle,
            mom.handle,
            math_ops.cast(self._learning_rate_tensor, grad.dtype),
            grad,
            indices,
            math_ops.cast(self._momentum_tensor, grad.dtype),
            use_locking=False,
            use_nesterov=use_nesterov,
        )
