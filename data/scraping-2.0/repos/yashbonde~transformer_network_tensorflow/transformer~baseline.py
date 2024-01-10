"""
baseline.py

This is the vanilla implementation of transformer network.

23.09.2019 - @yashbonde
"""

import tensorflow as tf
from types import SimpleNamespace

from .ops_util import get_opt
from .tf_layers import encoder_block, decoder_block, embed_sequence, positions_for, ff, noam_scheme, label_smoothing

from .common_layer_fns import shift_right_2d


def prepare_decoder_function(target, pad=None):
    decoder_input = shift_right_2d(target, pad)
    return decoder_input


def decoder_fn(config, dec_out, enc_out, encoder_pad_mask, decoder_pad_mask):
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        for layer_idx in range(config.num_layers):
            dec_out = decoder_block(q = dec_out, k = enc_out, v = enc_out, enc_mask = encoder_pad_mask,
                dec_mask = decoder_pad_mask, scope = 'layer_{}'.format(layer_idx), config = config)
    return dec_out


def encoder_fn(config, enc_out, encoder_pad_mask):
    with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
        for layer_idx in range(config.num_layers):
            enc_out = encoder_block(q = enc_out, ext_mask = encoder_pad_mask, scope = 'layer_{}'.format(layer_idx),
                config = config)
    return enc_out


def transformer(config, encoder_placeholder, target_placeholder=None, training=False):
    """
    Function for making and training baseline model. This style of code is heavily
    inspired from OpenAI's code.

    encoder_placeholder: placeholder to encoder stacks, tensor at the bottom of encoder
        [batch_size, input_length]
    target_placeholder: placeholder with target values, tensor at the top of decoder
        [batch_size, target_length]
    """
    encoder_inp = encoder_placeholder
    # decoder_inp = prepare_decoder_function(target_placeholder)  # TODO: Fix this
    decoder_inp = target_placeholder[:, :-1]

    with tf.variable_scope('embed'):
        '''
        NOTE: There are some differences from the method that was used by authors in their paper.
              They used the sinusoidal method for making positional embedding. This was basically a
              simple hack by creating multiple switch on/somewhere in the middle/off system. OpenAI
              instead used a much simpler method where they created an embedding matrix for positions
              along with one for context.

              To do that we use the `positions_for` method, which gives us a range according to input
              length of tensor.

              There are many different thing that we can actually this kind of embdding for
                - Additional of extra features such as POS /+ NER tags

              I found during my experiments that simply adding more and more tensors often improves
              the results. My hypothesis is that since we have a fixed embedding dimension for
              different features such as context and position. Adding more simply means that we keep
              morphing it's structure. This might cause difficulty initially however once the
              embedding matrices are trained this will be compensated. 
        '''

        # we start with embedding the input sequence
        (enc_con_emb, dec_con_emb), con_emb_matrix = embed_sequence(encoder_inp, decoder_inp, in_dim=config.vocab_size,
                                                    out_dim=config.embedding_dim, scope='context')

        (enc_pos_emb, dec_pos_emb), pos_emb_matrix = embed_sequence(
            positions_for(encoder_inp, past_length=0),
            positions_for(decoder_inp, past_length=0),
            in_dim=max(config.cntx_len, config.max_decode_length),
            out_dim=config.embedding_dim,
            scope='position')

        # appropriate value normalisation and getting masks
        enc_con_emb *= config.embedding_dim ** 0.5
        dec_con_emb *= config.embedding_dim ** 0.5
        encoder_pad_mask = tf.math.equal(encoder_inp, config.pad_id, name = 'encoder_pad_masking')
        decoder_pad_mask = tf.math.equal(decoder_inp, config.pad_id, name = 'decoder_pad_masking')

        # add the two embeddings
        enc_out = tf.layers.dropout(enc_con_emb + enc_pos_emb, 0.3, training = training)
        dec_out = tf.layers.dropout(dec_con_emb + dec_pos_emb, 0.3, training = training)

        # print('main_model > enc_out: {}'.format(enc_out))
        # print('main_model > dec_out: {}'.format(dec_out))
        
    # now we make the model, this is simple matter of calling the layers --> calling encoder and decoder functions
    enc_out = encoder_fn(config = config, enc_out = enc_out, encoder_pad_mask = encoder_pad_mask)
    dec_out = decoder_fn(config = config, dec_out = dec_out, enc_out = enc_out,
        encoder_pad_mask = encoder_pad_mask, decoder_pad_mask = decoder_pad_mask)

    if config.use_inverse_embedding:
        # use the same embedding for input and output
        pred_logits = tf.matmul(dec_out, con_emb_matrix, transpose_b = True)
        fproj_w, fproj_b = con_emb_matrix, None
    else:
        # use different embedding
        pred_logits, fproj_w, fproj_b = ff(dec_out, 'final_projection', config.vocab_size, return_param = True)

    pred_seq = tf.argmax(pred_logits, axis = 2) # [bs, seqlen]

    if training:
        # calculate loss --> use the just below one for using without label smoothing
        # print('\n\n\ndecoder_inp: {}\ntarget: {}\npred_logits: {}\n\n'.format(decoder_inp, target_placeholder[:, 1:], pred_logits))
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = target_placeholder[:, 1:],
                logits = pred_logits,
            )
        )

        # train ops with gradient clipping
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(config.lr, global_step, 4000)
        opt = get_opt(config.opt)(lr)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in opt.compute_gradients(loss)]
        train_step = opt.apply_gradients(capped_gvs, global_step)
        train_step = get_opt(config.opt)(lr).minimize(loss, global_step)

        # simple optimizer
        # train_step = get_opt(config.opt)(config.lr).minimize(loss)

        # summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('lr', lr)

        ret = SimpleNamespace(
            context_embedding = con_emb_matrix,
            position_embedding = pos_emb_matrix,
            pred_seq = pred_seq,
            pred_logits = pred_logits,
            loss = loss,
            train_step = train_step,
            encoder_embedding = enc_out,
            encoder_pad_mask = encoder_pad_mask,
            decoder_pad_mask = decoder_pad_mask,
            fproj_w = fproj_w,
            fproj_b = fproj_b
        )

        return ret

    else:
        return pred_logits, pred_seq
