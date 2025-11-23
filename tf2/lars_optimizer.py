# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""

import re

import tensorflow.compat.v2 as tf

EETA_DEFAULT = 0.001


class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer"):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
    """
    # Pass learning_rate to base class (required in TF 2.18+)
    # Pass weight_decay=None to base class to disable its weight decay
    # LARS handles weight decay internally in _resource_apply_dense
    super(LARSOptimizer, self).__init__(learning_rate=learning_rate, name=name, weight_decay=None)
    
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay
    
    # Store momentum variables in a simple list (serializable)
    self._momentum_list = []

  def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False):
    """Apply gradients to variables.
    
    Completely override to bypass base class weight decay.
    """
    grads_and_vars = list(grads_and_vars)
    if len(grads_and_vars) == 0:
      return tf.no_op()
    
    grads, trainable_variables = zip(*grads_and_vars)
    
    # Prepare aggregation state
    with tf.name_scope(self.name):
      # Build if not already built
      with tf.init_scope():
        self.build(trainable_variables)
      
      # Increment iterations
      self.iterations.assign_add(1)
      
      # Call _distributed_apply
      if tf.distribute.has_strategy():
        distribution = tf.distribute.get_strategy()
        return self._distributed_apply(
            distribution, grads_and_vars, name, apply_state=None)
      else:
        # Non-distributed case
        update_ops = []
        for grad, var in grads_and_vars:
          if grad is not None:
            update_ops.append(self._resource_apply_dense(grad, var))
        return tf.group(*update_ops)

  def build(self, var_list):
    # Call parent build
    super().build(var_list)
    if hasattr(self, "_built") and self._built:
      return
    self._built = True
    # Create momentum variables and store in list
    self._momentum_list = []
    for var in var_list:
      with tf.device(var.device):
        momentum_var = tf.Variable(
          tf.zeros_like(var),
          trainable=False,
          name=f"{self.name}/momentum/{var.name.split(':')[0].replace('/', '_')}"
        )
        self._momentum_list.append(momentum_var)
  
  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    """`apply_gradients` using a `DistributionStrategy`.
    
    Override to bypass base class weight decay since LARS handles it internally.
    """
    def apply_grad_to_update_var(var, grad):
      """Apply gradient to variable."""
      if grad is None:
        return tf.no_op()
      return self._resource_apply_dense(grad, var, apply_state=apply_state)
    
    eagerly_outside_functions = hasattr(distribution.extended,
                                       "_retrace_functions_for_each_device")
    update_ops = []
    with tf.name_scope(name or self.name):
      for grad, var in grads_and_vars:
        if grad is not None:
          with distribution.extended.colocate_vars_with(var):
            with tf.name_scope("update" if eagerly_outside_functions else
                             "update_" + var.op.name):
              update_op = distribution.extended.update(
                  var, apply_grad_to_update_var, args=(grad,), group=False)
              update_ops.append(update_op)
      
      any_symbolic = any(isinstance(i, tf.Operation) for i in update_ops)
      if not tf.executing_eagerly() or any_symbolic:
        with tf.control_dependencies(update_ops):
          return self._iterations.assign_add(1, read_value=False)
      return self._iterations.assign_add(1)

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    # Find momentum variable by matching parameter name
    v = None
    param_base_name = param.name.split(':')[0]
    for momentum_var in self._momentum_list:
      if param_base_name.replace('/', '_') in momentum_var.name:
        v = momentum_var
        break
    
    if v is None:
      # Create momentum variable if not found (shouldn't happen if build was called)
      with tf.device(param.device):
        v = tf.Variable(tf.zeros_like(param), trainable=False)
        self._momentum_list.append(v)

    if self._use_weight_decay(param_name):
      grad += self.weight_decay * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      # Convert to string if it's a tensor or variable
      param_name_str = param_name if isinstance(param_name, str) else str(param_name)
      for r in self.exclude_from_weight_decay:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name_str) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      # Convert to string if it's a tensor or variable
      param_name_str = param_name if isinstance(param_name, str) else str(param_name)
      for r in self.exclude_from_layer_adaptation:
        # TODO(srbs): Try to avoid name based filtering.
        if re.search(r, param_name_str) is not None:
          return False
    return True

  def _fallback_apply_state(self, var_device, var_dtype):
    """Fallback apply state when not using TPU."""
    # Handle both float learning rate and learning rate schedule
    # Use self.learning_rate (base class property) instead of self._learning_rate
    if callable(self.learning_rate):
      # Learning rate schedule - will be called with iterations
      lr = self.learning_rate(self.iterations)
    else:
      lr = self.learning_rate
    return {"lr_t": tf.cast(lr, var_dtype)}

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
    })
    return config
