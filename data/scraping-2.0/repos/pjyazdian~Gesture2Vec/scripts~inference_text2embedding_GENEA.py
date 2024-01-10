import argparse
import math
import os
import pickle
import pprint
import random
from pathlib import Path
from datetime import datetime

import librosa
import numpy as np
import time

import torch
from scipy.signal import savgol_filter
import joblib as jl

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

from inference_Autoencoder import smoothing_function

from trinity_data_to_lmdb import process_bvh as process_bvh_trinity
from twh_dataset_to_lmdb import process_bvh as process_bvh_twh
from twh_dataset_to_lmdb import process_bvh_rot_only as process_bvh_rot_only_twh
from twh_dataset_to_lmdb import process_bvh_rot_only_Taras as process_bvh_rot_only_Taras
from inference_DAE import feat2bvh



import matplotlib.pyplot as plt
import openai
import os
import pickle




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_TYPE = 'TWH'


infer_from_high_quality = False
high_quality_model_adress = '../output/autoencoder/toturial/ablation-study/6_Vanilla_DerD/VQ-DVAE_ablation1_checkpoint_020.bin'
high_quality_model_adress = '../output/IROS_2/AI2_11_HQ/VQVAE_checkpoint_020.bin' # 11-->20f
high_quality_model_adress = '../output/GENEA/DAE/train_DAE_H20/DAE_H20_checkpoint_050.bin'
high_quality_model_DAE_adress = '../output/IROS_2/DAE_p2/DAE_H40_checkpoint_020.bin'

Inference_audio = False
openai.api_key = "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"

def GPT_3_caller( str_input, pickle_file_loaded):
    if pickle_file_loaded != None:
        for i in range(len(pickle_file_loaded['sample_words_list'])):
            if str_input == pickle_file_loaded['sample_words_list'][i]:
                print("Embedding found:", str_input)
                return pickle_file_loaded['GPT_3_Embedding_list'][i]

    request = openai.Embedding.create(input=str_input, engine="text-similarity-ada-001")
    embedding = request['data'][0]['embedding']
    print("!!!!!!!!!!!!!!!!!!!Embedding requested:", str_input)
    return embedding

def plot_attentions(attention_list):
    try:
        for i in range(len(attention_list)):
            attention_list[i] = attention_list[i].squeeze().squeeze().cpu().detach().numpy()
        attention_list = np.array(attention_list)
        plt.imshow(attention_list)
        plt.show()
    except:
        return


def generate_gestures(args, pose_decoder, lang_model,
                      words, audio_raw, audio_sr, poses,
                      DAE, Autoencoder, HAutoencoder, GPT3_cach, GPT3_Address):

    # Todo----------------------------------------------
    # Find nearest neighbour

    # loaded = pickle.load(open('../output/clustering_results/org_latent_clustering_data.bin', 'rb'))

    import os
    loaded = pickle.load(open(os.path.dirname(args.autoencoder_checkpoint)
                                       + '/clusters/org_latent_clustering_data.bin', 'rb'))

    kmeanmodel = pickle.load(open(os.path.dirname(args.autoencoder_checkpoint)
                                       + '/clusters/kmeans_model.pk', 'rb'))
    # clustering_transforms = pickle.load(open(os.path.dirname(args.autoencoder_checkpoint)
    #                                               + '/clusters/transforms.pkl', 'rb'))



    loaded = np.hstack(loaded)
    # all_dataset_latents = []
    # count = 0
    # for item in loaded:
    #     count = count+1
    #     if count>2000:
    #         break
    #     current_latent_rnn = item['latent_rnn']
    #     current_latent_rnn = np.hstack(current_latent_rnn)
    #     all_dataset_latents.append(current_latent_rnn)
    # all_dataset_latents = np.array(all_dataset_latents)
    # Todo----------------------------------------------

    # Todo------------------from get sample function-----
    # loaded  = pickle.load(open('../output/clustering_results/org_latent_clustering_data.bin','rb'))
    loaded = np.hstack(loaded)
    # clustering_transforms = pickle.load(open('../output/transforms.pkl', 'rb'))

    if Autoencoder[1].vq: #pose_decoder.decoder.discrete_representation:
        number_of_clusters = pose_decoder.pose_dim
        predicted_labels = np.zeros(len(loaded))

        cluster_indexed = dict()
        for i in range(number_of_clusters):
            cluster_indexed[i] = []

        for i in range(len(loaded)):
            current_cluster_index = loaded[i]['quantized_indices'][0]
            predicted_labels[i] = current_cluster_index
            cluster_indexed[current_cluster_index].append(i)
    else:
        number_of_clusters = 300 # for kmeans
        data_latent_rnn = []
        for i in range(len(loaded)):
            current_latent_rnn = loaded[i]['latent_rnn']
            current_latent_rnn = np.hstack(current_latent_rnn)
            data_latent_rnn.append(current_latent_rnn)
        data_latent_rnn_ndarray = np.array(data_latent_rnn)

        # Applying transformations and clustering.
        # transfromed = clustering_transforms['scalar'].transform(data_latent_rnn_ndarray)
        predicted_labels = kmeanmodel.predict(data_latent_rnn_ndarray)

        print("Loop started")

        cluster_indexed = dict()
        for c_index in range(512):
            selected_list_indexes = []
            for ix in range(len(predicted_labels)):
                if predicted_labels[ix] == c_index:
                    selected_list_indexes.append(ix)
                    break # one sample from each cluster
            cluster_indexed[c_index] = selected_list_indexes
        print("Loop Done")

    # Todo----------------------------------------------



    out_list = []
    clip_length = words[-1][2]

    # pre seq
    pre_seq = torch.zeros((args.n_pre_poses, pose_decoder.sentence_frame_length // pose_decoder.n_frames))#pose_decoder.pose_dim))
    pre_seq = torch.zeros(
        (1, pose_decoder.sentence_frame_length // pose_decoder.n_frames))  # pose_decoder.pose_dim))
    pre_seq = pre_seq.long()
    # if seed_seq is not None:
    #     pre_seq[0, :, :] = torch.Tensor(seed_seq[0:args.n_pre_poses])
    # else:
    #     mean_pose = args.data_mean
    #     mean_pose = torch.squeeze(torch.Tensor(mean_pose))
    #     pre_seq[0, :, :] = mean_pose.repeat(args.n_pre_poses, 1)

    # divide into inference units and do inferences
    # unit_time = args.n_poses / args.motion_resampling_framerate
    unit_time = args.sentence_frame_length / args.motion_resampling_framerate
    # stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    # Todo: fix this    120
    stride_time = 4*args.subdivision_stride_sentence / args.motion_resampling_framerate

    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time)

    print('{}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time))
    # num_subdivision = min(num_subdivision, 59)  # DEBUG: generate only for the first N divisions
    # num_subdivision = min(num_subdivision, 20)  # DEBUG: generate only for the first N divisions

    out_poses = None
    start = time.time()
    all_frames_from_rnn = None
    all_frames_from_original = None

    pre_dim = 20 # DAE[4] #Todo: fix this later
    decoder_input = torch.zeros((1, pre_dim)).to(device).unsqueeze(0)
    decoder_input = decoder_input.squeeze(0).unsqueeze(0)
    decoder_input = decoder_input.transpose(0, 1)

    input_pre_seq = None

    cheat = None

    GPT_3_Embedding_list = []
    GPT_3_STR_list = []
    for i in range(0, num_subdivision):
        # start_time = i * stride_time
        start_time = i * unit_time
        end_time = start_time + unit_time

        # Prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        str_input = ""
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            word_indices[w_i + 1] = lang_model.get_word_index(word[0]) #word[0])
            str_input += word[0] + ' '
        print(' ({}, {})'.format(start_time, end_time))
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)


        GPT_3_features = GPT_3_caller(str_input, GPT3_cach)
        GPT_3_features = np.array(GPT_3_features)
        GPT_3_Embedding_list.append(GPT_3_features)
        GPT_3_STR_list.append(str_input)
        # Discretization for the output side:
        GPT_3_features = torch.from_numpy(GPT_3_features).float().unsqueeze(0).to(device)
        # Prepare audio
        if Inference_audio:
            audio_start = math.floor(start_time * audio_sr)
            audio_end = int(audio_start + unit_time * audio_sr)

            try:
                in_audio = (audio_raw[audio_start:audio_end])
                audi_smaple_rate = 16000
                mels = []
                for audio_sub in range(len(in_audio) // audi_smaple_rate):
                    audio_chunk = in_audio[
                                  audio_sub * audi_smaple_rate: (audio_sub + 1) * audi_smaple_rate]
                    signal = librosa.feature.melspectrogram(y=audio_chunk, sr=audi_smaple_rate)
                    signal = librosa.power_to_db(signal, ref=np.max)
                    mels.append(signal)
                    # signal = librosa.amplitude_to_db(signal)
                in_audio = np.array(mels)
                in_audio = torch.from_numpy(in_audio).float().to(device)
                in_audio = in_audio.unsqueeze(0)
            except:
                print("xx")
        else:
            in_audio = None

        # prepare pre seq
        # if i > 0:
        #     pre_seq[0, :, :] = out_poses.squeeze(0)[-args.n_pre_poses:]
        #     pre_seq[:,0] = out_sentence_latent[].argmax(1)
        # handled in chet parameter
        pre_seq = pre_seq.to(device)

        # inference
        words_lengths = torch.LongTensor([in_text.shape[1]]).to('cpu')

        target_sentence_latent = ""

        out_sentence_latent, attention_list = pose_decoder(in_text, words_lengths, in_audio, pre_seq, GPT_3_features, cheat)
        plot_attentions(attention_list)
        # out_seq = out_poses[0, :, :].data.cpu().numpy()
        # todo: wwwcheck:
        best_guess = out_sentence_latent.argmax(2)
        cheat = best_guess[:, -1]


        # 2. Go through autoencoder
        Autoencoder_args, Autoencoder_generator, Autoencoder_loss_fn,\
        Autoencoder_lang_model, Autoencoder_out_dim = Autoencoder

        DAE_args, DAE_generator, DAE_loss_fn, \
        DAE_lang_model, DAE_out_dim = DAE
        DAE_generator.train(False)

        # High quality
        if infer_from_high_quality:
            HAutoencoder_args, HAutoencoder_generator, HAutoencoder_loss_fn, \
            HAutoencoder_lang_model, HAutoencoder_out_dim = HAutoencoder
            HAutoencoder_generator.train(False)
        reconstructed_rnn = torch.zeros(Autoencoder_args.n_poses,
                                        1,
                                        Autoencoder_generator.decoder.output_size)\
            .to(out_sentence_latent.device)
        out_sentence_latent = out_sentence_latent.squeeze(0)

        # Todo: moved to the top for smoothness
        # decoder_input = torch.zeros((1, 82)).to(device).unsqueeze(0)
        # decoder_input = decoder_input.squeeze(0).unsqueeze(0)
        # decoder_input = decoder_input.transpose(0,1)


        for latent_counter in range((out_sentence_latent.size(0))):

            # if Autoencoder_generator.text2_embedding_discrete:
            if True:
                decoder_hidden = out_sentence_latent[latent_counter].argmax(0)
                check1 = out_sentence_latent[latent_counter].to('cpu').detach().numpy()
                check2 = decoder_hidden.to('cpu').detach().numpy()
                print("\n--Cluster--> ",check2)
                original_gestures =\
                    get_sample_from_dataset(args, loaded=loaded,
                                            c_index=decoder_hidden,
                                            cluster_indexed=cluster_indexed,
                                            seq_length=Autoencoder_args.n_poses)



                # latent_lin =  latent_lin.to(device)
                # decoder_hidden = decoder_hidden.to(device).float()
                # decoder_hidden = decoder_hidden.reshape(Autoencoder_args.n_layers, -1)


                target = (original_gestures)
                target = target.to(device).float()
                if DAE_generator.encoder == None:
                    encoded = target
                else:
                    encoded = DAE_generator.encoder(target)

                # if torch.eq(encoded , latent_lin):
                # encoded = latent_lin

                use_derivitive = False
                if use_derivitive:
                    diff = [(encoded[n, :] - encoded[n - 1, :]) for n in range(1, encoded.shape[0])]
                    diff.insert(0, torch.zeros_like(encoded[0, :]))
                    encoded = torch.hstack((encoded, torch.stack(diff)))
                input_seq = encoded
                if input_pre_seq == None:
                    input_pre_seq = encoded[i] #?
                input_seq = torch.unsqueeze(input_seq, 0)
                input_seq = input_seq.transpose(0, 1)

                if infer_from_high_quality:
                    encoder_outputs, encoder_hidden = HAutoencoder_generator.encoder(input_seq, None)
                    decoder_hidden = (encoder_hidden[:HAutoencoder_generator.decoder.n_layers])
                    #Todo: not Variational inference
                else:
                    encoder_outputs, encoder_hidden = Autoencoder_generator.encoder(input_seq, None)
                    decoder_hidden = (encoder_hidden[:Autoencoder_generator.decoder.n_layers])

            else:
                decoder_hidden = out_sentence_latent[latent_counter]
                decoder_hidden = decoder_hidden.reshape(Autoencoder_args.n_layers, -1)

            # # Find neaarest neighbour
            # value = decoder_hidden.clone().cpu().detach().numpy()
            # idx = np.abs(np.sum(all_dataset_latents - value)).argmin()
            # decoder_hidden = all_dataset_latents[idx]
            # decoder_hidden = torch.from_numpy(decoder_hidden).to(device)


            decoder_hidden = decoder_hidden.reshape(Autoencoder_args.n_layers, -1)
            decoder_hidden = decoder_hidden.unsqueeze(1)
            # decoder_hidden.transpose(0, 1)
            # output_seq = torch.zeros((30,82))
            # output_seq = torch.unsqueeze(output_seq, 0)
            # output_seq = output_seq.transpose(0, 1)

            # decoder_input = torch.zeros_like(decoder_input).to(device)  #moved to the top for smoothness reasons

            for t in range(0, Autoencoder_args.n_poses):
                # if not Autoencoder_generator.autoencoder_conditioned:
                #     # Todo: why while it is a conditional network?     This was from autoencoder
                #     decoder_input = torch.zeros_like(decoder_input).to(device)
                if infer_from_high_quality:
                    decoder_output, decoder_hidden, _ = HAutoencoder_generator.decoder(None, decoder_input, decoder_hidden, None,
                                                                     None)
                else:
                    decoder_output, decoder_hidden, _ = Autoencoder_generator.decoder(None, decoder_input,
                                                                                       decoder_hidden, None,
                                                                                       None)

                # Todo: fix the smoothness problem: Give the chance of being in the right current
                # for iq in range(1):
                #     decoder_output, decoder_hidden, _ = Autoencoder_generator.decoder(None, decoder_input,
                #                                                                       decoder_hidden, None,
                #                                                                       None)


                try:
                    reconstructed_rnn[t] = decoder_output
                except:
                    print()

                if t < Autoencoder_generator.n_pre_poses and False:
                    decoder_input = output_seq[t]  # next input is current target
                else:
                    decoder_input = decoder_output  # next input is current prediction
                # Todo: fix this regarding decoder trained autoencoder
                # if not Autoencoder_generator.autoencoder_conditioned:
                #     decoder_input = torch.zeros_like(decoder_input)


            reconstructed_rnn = encoded.unsqueeze(1)

            if all_frames_from_rnn == None:
                all_frames_from_rnn = reconstructed_rnn.transpose(0,1)
            else:
                all_frames_from_rnn = torch.cat((all_frames_from_rnn, reconstructed_rnn.transpose(0, 1)), 1)
                # all_frames_from_rnn = torch.cat((all_frames_from_rnn, encoded.transpose(0, 1)), 1)

            if all_frames_from_original == None:
                all_frames_from_original = target.unsqueeze(0)
            else:
                all_frames_from_original = torch.cat((all_frames_from_original, target.unsqueeze(0)), 1)

    #     Todo: decode DAE
    all_frames_from_rnn = torch.squeeze(all_frames_from_rnn, 0)
    if infer_from_high_quality:
        use_derivitive = HAutoencoder_args.use_derivitive == 'True'
    else:
        use_derivitive = Autoencoder_args.use_derivitive == 'True'
    if use_derivitive:
        all_frames_from_rnn = all_frames_from_rnn[:, 0: all_frames_from_rnn.shape[1] // 2]
    if DAE_generator.encoder == None:
        reconstructed_seq_DAE = all_frames_from_rnn
    else:
        reconstructed_seq_DAE = DAE_generator.decoder(all_frames_from_rnn)

    # reconstructed_seq_DAE = torch.squeeze(reconstructed_seq_DAE, 2)
    reconstructed_seq_DAE = reconstructed_seq_DAE.to('cpu')
    reconstructed_seq_DAE = reconstructed_seq_DAE.detach().numpy()

    # Todo: just to check sanity
    # reconstructed_seq_DAE = all_frames_from_original.squeeze().detach().cpu().numpy()



    print('Avg. inference time: {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results


    print("Saving OPENAI GPT3 calss")
    object_to_save = {'sample_words_list': GPT_3_STR_list,
                      'GPT_3_Embedding_list': GPT_3_Embedding_list}
    pickle.dump(object_to_save, open(GPT3_Address, 'wb'))

    return np.array(reconstructed_seq_DAE)


def get_sample_from_dataset(args, loaded, c_index, cluster_indexed,  seq_length=30):

    #Todo: fox to use quantization index

    try:
        selected_list_indexes = cluster_indexed[int(c_index.cpu().detach().numpy())]
    except:
        print("22")
    if selected_list_indexes == []:
        print("**** Empty list occuered!!!")

    while selected_list_indexes == []:
        rnd = random.randint(0, len(cluster_indexed) - 1)
        selected_list_indexes = cluster_indexed[rnd]
    while True:
        rnd = random.randint(0, len(selected_list_indexes)-1)
        if loaded[selected_list_indexes[rnd]]['latent_linear'].shape[0] == seq_length:
            break

    original = (loaded[selected_list_indexes[rnd]]['original'])
    print("Chosen:", selected_list_indexes[rnd])


    # Since it has been saved as raw data in Clustering.py
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = ((original) - mean) / std
    out_poses = torch.from_numpy(out_poses)



    return out_poses

# def main(checkpoint_path, transcript_path, rep_learning_checkpoint, autoencoder_checkpoint):
    # GAN = False
    # if GAN:
    #     args, generator, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
    #         checkpoint_path, device, what='text2embedding_gan')
    #     generator = generator.generator
    # else:
    #     args, generator, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
    #         checkpoint_path, device, what='text2embedding')
    #
    # DAE = utils.train_utils.load_checkpoint_and_model(
    #     rep_learning_checkpoint, device, what='DAE')
    # Autoencoder = utils.train_utils.load_checkpoint_and_model(
    #     autoencoder_checkpoint, device, what='autoencoder_vq')
    #
    # HAutoencoder = utils.train_utils.load_checkpoint_and_model(
    #     high_quality_model_adress, device, what='autoencoder')

def main(checkpoint_path, txt2embedding_model,
         transcript_path, audio_path, poses_path,
         DAE, Autoencoder, HAutoencoder):

    args, generator, loss_fn, lang_model, out_dim = txt2embedding_model

    pprint.pprint(vars(args))
    save_path = '../output/infer_sample'
    save_path = os.path.dirname(checkpoint_path) + '/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    big_infer = False
    if not big_infer:
        transcript = SubtitleWrapper(transcript_path).get()
        # load audio
        audio_raw, audio_sr = librosa.load((audio_path),
                                mono=True, sr=16000, res_type='kaiser_fast')

        if DATASET_TYPE == 'Trinity':
            poses, poses_mirror = process_bvh_trinity(poses_path)
        elif DATASET_TYPE == 'TWH':
            poses = process_bvh_rot_only_Taras(poses_path)
            poses_mirror = poses

    else:
        adress = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq00"
        offset = 0
        transcript_list = []
        for i in range(1, 5):
            current_transcript = SubtitleWrapper(adress + str(i) + ".json" ).get()
            for word in current_transcript:
                word['start_time'] = "{:.2f}".format(offset + float(word['start_time'][0:-1]) ) + 's'
                word['end_time'] = "{:.2f}".format(offset + float(word['end_time'][0:-1]) ) + 's'
            offset = float(current_transcript[-1]['end_time'][0:-1])
            transcript_list.append(current_transcript)
        transcript = [item for subist in transcript_list for item in subist]

    GPT3_cach_address = str(transcript_path)[:-5] + '.gpt'

    if os.path.exists(GPT3_cach_address):
        loaded_GPT = pickle.load(open(GPT3_cach_address,'rb'))
    else:
        loaded_GPT = None

    word_list = []
    for wi in range(len(transcript)):
        word_s = float(transcript[wi]['start_time'][:-1])
        word_e = float(transcript[wi]['end_time'][:-1])
        word = transcript[wi]['word']

        word = normalize_string(word)
        if len(word) > 0:
            word_list.append([word, word_s, word_e])

    # inference

    # Plot motion cluster embedding
    m_embedding_weights = generator.decoder.decoder.embedding.weight.cpu().detach().numpy()
    plt.imshow(m_embedding_weights)
    plt.title("Motion cluster embeddings")
    plt.show()


    out_poses = generate_gestures(args, generator, lang_model,
                                  word_list, audio_raw, audio_sr, poses,
                                  DAE, Autoencoder, HAutoencoder,
                                  loaded_GPT, GPT3_cach_address)

    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = np.multiply(out_poses, std) + mean

    # make a BVH
    filename_prefix = '{}'.format(transcript_path.stem)
    if DATASET_TYPE=='Trinity':
        make_bvh(save_path, filename_prefix, out_poses)
    elif DATASET_TYPE == 'TWH':
        feat2bvh(save_path, filename_prefix, out_poses)

def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline = jl.load('../resource/data_pipe.sav')

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 25, 5)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()


    # New smoothing approach in 3D space:
    out_euler = smoothing_function('spline', out_euler)


    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + '.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


if __name__ == '__main__':

    '''
      ../output/autoencoder/toturial/ICLR_text2embedding/VQ-DVAE_ablation1_checkpoint_020.bin
        /local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
        ../output/DAE_old/train_DAE_H41/rep_learning_DAE_H41_checkpoint_020.bin
        ../output/autoencoder/toturial/4th/VQ-DVAE_ablation1_checkpoint_015.bin
    
    
        ../output/IROS/Proposed4/text2mbedding/VQ-DVAE_ablation1_checkpoint_010.bin
/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
../output/IROS/DAE/DAE_H40_checkpoint_020.bin
../output/IROS/Proposed4/VQVAE_checkpoint_020.bin


../output/GENEA/VQ-VAE/text2mbedding/VQ-DVAE_ablation1_checkpoint_050.bin
Nothing
../output/GENEA/DAE/train_DAE_H20/DAE_H20_checkpoint_050.bin
../output/GENEA/VQ-VAE/VQVAE_checkpoint_020.bin
    
    
       '''
    random.seed(datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("transcript_path", type=Path)
    parser.add_argument("rep_learning_checkpoint", type=Path)
    parser.add_argument("autoencoder_checkpoint", type=Path)

    args = parser.parse_args()

    pre_transcript_path = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/"
    pre_audio_path = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Audio/"
    pre_bvh_pth = "/local-scratch/pjomeyaz/rosie_gesture_benchmark/cloned/Clustering/must/GENEA/Co-Speech_Gesture_Generation/dataset/dataset_v1/val/bvh/"
    # From inside the main to increase running time
    GAN = False
    if GAN:
        txt2embedding_model = utils.train_utils.load_checkpoint_and_model(
            args.ckpt_path, device, what='text2embedding_gan')
        generator = generator.generator
    else:
        txt2embedding_model = utils.train_utils.load_checkpoint_and_model(
            args.ckpt_path, device, what='text2embedding')

    DAE = utils.train_utils.load_checkpoint_and_model(
        args.rep_learning_checkpoint, device, what='DAE')
    Autoencoder = utils.train_utils.load_checkpoint_and_model(
        args.autoencoder_checkpoint, device, what='autoencoder_vq')

    HAutoencoder = None
    if infer_from_high_quality:
        HAutoencoder = utils.train_utils.load_checkpoint_and_model(
            high_quality_model_adress, device, what='autoencoder_vq')
        HDAE = utils.train_utils.load_checkpoint_and_model(
            HAutoencoder[0].rep_learning_checkpoint, device, what='DAE')

        DAE = HDAE

    for i in reversed(range(1, 2)): #11)):
        transcript_path = pre_transcript_path + "TestSeq" + str(i).zfill(3) + ".json"
        transcript_path = Path(transcript_path)
        audio_path = pre_audio_path + "TestSeq" + str(i).zfill(3) + ".wav"
        poses_path = pre_bvh_pth + "val_2022_v1_" + str(i).zfill(3) + ".bvh"
        # plot_loss(args.ckpt_path_Autoencode)
        # main(args.ckpt_path, args.transcript_path, args.rep_learning_checkpoint, args.autoencoder_checkpoint)
        # main(args.ckpt_path, transcript_path, args.rep_learning_checkpoint, args.autoencoder_checkpoint)
        main(args.ckpt_path, txt2embedding_model,
             transcript_path, audio_path, poses_path,
             DAE, Autoencoder, HAutoencoder)
