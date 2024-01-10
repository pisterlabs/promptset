from summa.module import OptimusModule
from summa import get_args
import torch
import summa.mpu as mpu
from summa.mpu import LayerNorm_summa
from summa.model.fused_softmax import FusedScaleMaskSoftmax
from summa.model.fused_bias_gelu import bias_gelu_impl
from summa.model.utils import openai_gelu, erf_gelu
import math
import torch.nn.functional as F

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     q: summa dimension. We have p = q^2
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [b/q, s, h/q] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""

class ParallelMLP(OptimusModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method, skip_bias_add=True):
        super(OptimusModule, self).__init__()
        args = get_args()
        self.skip_bias_add = skip_bias_add

        h4h_forward_buffer = mpu.get_h4h_forward_buffer()
        fhh_forward_buffer = mpu.get_fhh_forward_buffer()
        backward_buffer = mpu.get_backward_buffer()
        parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()

        self.dense_h_to_4h = mpu.SUMMALinear(
            args.hidden_size, 4*args.hidden_size,
            bias_flag=True, init_method=init_method,
            skip_bias_add=self.skip_bias_add,
            forward_buffer=h4h_forward_buffer,
            backward_buffer=backward_buffer,
            parameter_gradient_buffer=parameter_gradient_buffer)

        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        self.dense_4h_to_h = mpu.SUMMALinear(
            4*args.hidden_size, args.hidden_size,
            bias_flag=True, init_method=output_layer_init_method,
            skip_bias_add=self.skip_bias_add,
            forward_buffer=fhh_forward_buffer,
            backward_buffer=backward_buffer,
            parameter_gradient_buffer=parameter_gradient_buffer)

    def forward(self, hidden_states):
        # hidden_states: [b/q, s, h/q]
        # [b/q, s, 4h/q]
        if self.skip_bias_add:
            intermediate_parallel, bias = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias)
        else:
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
        # [b/q, s, h/q]

        if self.skip_bias_add:
            output, bias = self.dense_4h_to_h(intermediate_parallel)
            return output, bias
        else:
            output = self.dense_4h_to_h(intermediate_parallel)
            return output


class ParallelSelfAttention(OptimusModule):

    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number):
        super(ParallelSelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        self.hidden_size_per_partition = mpu.divide(
            args.hidden_size, args.summa_dim)
        self.hidden_size_per_attention_head = mpu.divide(
            args.hidden_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, args.summa_dim)

        QKV_forward_buffer = mpu.get_QKV_forward_buffer()
        QKV_dense_buffer = mpu.get_QKV_dense_buffer()
        backward_buffer = mpu.get_backward_buffer()
        parameter_gradient_buffer = mpu.get_parameter_gradient_buffer()

        self.query_key_value = mpu.SUMMALinear(
            args.hidden_size, 3*args.hidden_size,
            bias_flag=False, init_method=init_method,
            forward_buffer=QKV_forward_buffer,
            backward_buffer=backward_buffer,
            parameter_gradient_buffer=parameter_gradient_buffer)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            args.scaled_upper_triang_masked_softmax_fusion,
            self.attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        self.dense = mpu.SUMMALinear(
            args.hidden_size, args.hidden_size,
            bias_flag=True, init_method=output_layer_init_method,
            forward_buffer=QKV_dense_buffer,
            backward_buffer=backward_buffer,
            parameter_gradient_buffer=parameter_gradient_buffer,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask):
        # hidden_state: [b/q, s, h/q]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [b/q, s, h/q] --> [b/q, s, 3*h/q]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [b/q, s, n/q, 3*h/n]
        new_tensor_shape = mixed_x_layer.shape[:-1] +\
                           (self.num_attention_heads_per_partition,
                            3*self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(new_tensor_shape)
        # [b/q, n/q, s, 3*h/n]
        mixed_x_layer = mixed_x_layer.permute((0, 2, 1, 3))

        # [b/q, n/q, s, 3*h/n] -> 3 [b/q, n/q, s, h/n]
        (query_layer, key_layer, value_layer) = torch.split(
            mixed_x_layer, self.hidden_size_per_attention_head, dim=-1)

        # [b/q, n/q, s, s]
        matmul_result = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b/q, n/q, s, s]
        attention_probs = self.scale_mask_softmax(matmul_result,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # [b/q, n/q, s, h/n]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b/q, n/q, s, h/n] -> [b/q, s, n/q, h/n]
        context_layer = context_layer.permute((0, 2, 1, 3)).contiguous()
        # [b/q, s, n/q, h/n] -> [b/q, s, h/q]
        new_context_layer_shape = context_layer.shape[:2] + (self.hidden_size_per_partition, )
        context_layer = context_layer.view(new_context_layer_shape)

        output, bias = self.dense(context_layer)
        return output, bias


def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(OptimusModule):
    """
    A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    """
    def __init__(self, mask_attention_func, init_method,
                 output_layer_init_method, layer_number):
        super(ParallelTransformerLayer, self).__init__()
        args = get_args()

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.hidden_pp = mpu.divide(args.hidden_size, args.summa_dim)
        self.input_layernorm = LayerNorm_summa(self.hidden_pp)
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Self attention.
        self.attention = ParallelSelfAttention(
            mask_attention_func,
            init_method,
            output_layer_init_method,
            layer_number)
        self.hidden_dropout = args.hidden_dropout

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm_summa(self.hidden_pp)

        # MLP
        self.mlp = ParallelMLP(init_method, output_layer_init_method,
                               skip_bias_add=self.bias_dropout_fusion)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [b/q, s, h/q]
        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, attention_bias \
            = self.attention(layernorm_output, attention_mask)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        if self.bias_dropout_fusion:
            mlp_output, bias = self.mlp(layernorm_output)
        else:
            mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.bias_dropout_fusion:
            with torch.enable_grad():
                output = bias_dropout_add_func(
                    mlp_output,
                    bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            output = F.dropout(
                mlp_output,
                p=self.hidden_dropout,
                training=self.training) + residual

        return output


class ParallelTransformer(OptimusModule):
    def __init__(self, attention_mask_func,
                 init_method, output_layer_init_method):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = args.num_unique_layers
        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = args.param_sharing_style

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number)
        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_unique_layers)])

        # Print layer ordering.
        if self.num_layers != self.num_unique_layers:
            if torch.distributed.get_rank() == 0:
                print('> will be using the following layer ordering:')
                for i in range(self.num_layers):
                    print('   layer id: {:3d} --> unique layer id: '
                          '{:3d}'.format(i, self._get_layer_index(i)),
                          flush=True)

        # Final layer norm before output.
        self.hidden_pp = mpu.divide(args.hidden_size, args.summa_dim)
        self.final_layernorm = LayerNorm_summa(self.hidden_pp)

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers)
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        # Make sure memory is freed.
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def forward(self, hidden_states, attention_mask):
        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(
                hidden_states, attention_mask)
        else:
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(hidden_states, attention_mask)
        output = self.final_layernorm(hidden_states)
        return output