""" create data samples """
import lmdb
import math
import numpy as np
import pyarrow
from deepsegment import DeepSegment
import torch
import librosa
import utils
from tqdm import tqdm
import openai
import pickle
import os
from trankit import Pipeline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DataPreprocessor:
    def __init__(self, args, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, sentence_level=False):
        openai.api_key = "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"
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.sentence_level = sentence_level
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        self.out_lmdb_dir = out_lmdb_dir
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        self.ckpt_path_DAE = args.rep_learning_checkpoint
        self.ckpt_path_Autoencode = args.autoencoder_checkpoint

        if args.name != "DAE":
            self.DAE_frame_level = utils.train_utils.load_checkpoint_and_model(
                self.ckpt_path_DAE, device,'DAE')

        if self.sentence_level:
            self.rnn_representation = utils.train_utils.load_checkpoint_and_model(
                self.ckpt_path_Autoencode, device, 'autoencoder_vq')

        # create db for samples
        map_size = 1024 * 50  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.sentence_frame_length = args.sentence_frame_length
        self.audi_smaple_rate = 16000


    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)
        total_count = src_txn.stat()['entries']
        # sampling and normalization
        cursor = src_txn.cursor()
        counter = 0
        for key, value in tqdm(cursor):
            print("video ", counter, "of", total_count, '\n')
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)
                counter = counter + 1
            # if counter>5:
            #     print("Break the chain of loading huge data!!!!!!1")
            #     break

        # print stats




        with self.dst_lmdb_env.begin() as txn:
            print("Sample_counter", txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        clip_word_list = clip['words']

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []

        sample_audio_list_mels = []
        sample_audio_list_raws = []

        sentence_leve_latents_list = []
        GPT_3_STR_list = []
        GPT_3_Embedding_list = []

        if self.sentence_level:
            self.n_poses = self.sentence_frame_length
            self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * self.audi_smaple_rate)

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        # Sentence level preparation:

        '''
        self.sentence_level = False
        if self.sentence_level:
            print("Sentence level")
            
            str_transcript = ""
            for i_t in range(len(clip_word_list)):
                str_transcript = str_transcript + " " + clip_word_list[i_t][0]

            import nltk  # library
            # nltk.download()
            sentences = nltk.sent_tokenize(str_transcript)  # whole paragraph break into sentence.

            start_sentence_word_index = 0
            end_sentence_word_index = 0
            for sentence in sentences:
                x = sentence.replace('.', '').strip().split(' ')

                end_sentence_word_index = start_sentence_word_index + len(sentence.replace('.', '').strip().split(' ')) - 1  # to index
                #     process

                reconstructed_sentence = clip_word_list[start_sentence_word_index:end_sentence_word_index+1]

                start_idx = int( clip_word_list[start_sentence_word_index][1] * self.skeleton_resampling_fps )
                try:
                    fin_idx = int( clip_word_list[end_sentence_word_index][2] * self.skeleton_resampling_fps )
                except:

                    print()
                # ........................
                sample_skeletons = clip_skeleton[start_idx:fin_idx]
                subdivision_start_time = start_idx / self.skeleton_resampling_fps
                subdivision_end_time = fin_idx / self.skeleton_resampling_fps
                sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                            start_time=subdivision_start_time,
                                                            end_time=subdivision_end_time)
                if len(sample_words) < 2:
                    continue

                # raw audio
                audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
                audio_end = audio_start + self.audio_sample_length
                sample_audio = clip_audio_raw[audio_start:audio_end]

                motion_info = {'vid': vid,
                               'start_frame_no': start_idx,
                               'end_frame_no': fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time}

                sample_skeletons_list.append(sample_skeletons)
                sample_words_list.append(sample_words)
                sample_audio_list.append(sample_audio)
                aux_info.append(motion_info)



                start_sentence_word_index = end_sentence_word_index + 1
                print()
            '''
        # Loading cached GP3 requestes
        address = self.out_lmdb_dir + '_' + vid + '.gpt'
        if os.path.exists(self.out_lmdb_dir + '_' + vid + '.gpt'):
            loaded_gpt3 = pickle.load(open(address, 'rb'))
        else:
            loaded_gpt3 = None

        for i in tqdm(range(num_subdivision)):



            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time)


            if len(sample_words) < 4:
                continue

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]

            mel_chunks = []
            raw_chunks = []
            for audio_sub in range(self.audio_sample_length//self.audi_smaple_rate):
                audio_chunk = sample_audio[audio_sub*self.audi_smaple_rate: (audio_sub+1)*self.audi_smaple_rate]
                signal = librosa.feature.melspectrogram(y=audio_chunk, sr=self.audi_smaple_rate)
                signal = librosa.power_to_db(signal, ref=np.max)
                mel_chunks.append(signal)
                # raw_chunks.append(audio_chunk)
                raw_chunks.append(0)
                # signal = librosa.amplitude_to_db(signal)







            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_words_list.append(sample_words)

            sample_audio_list_mels.append(mel_chunks)
            sample_audio_list_raws.append(raw_chunks)

            aux_info.append(motion_info)
            if self.sentence_level:

                # For the input side:
                # GPT-3 Embeddomg
                str_input = ""
                for word in sample_words:
                    str_input += ' ' + word[0]

                GPT_3_features = self.GPT_3_caller(str_input, loaded_gpt3)
                GPT_3_Embedding_list.append(GPT_3_features)
                GPT_3_STR_list.append(str_input)
                # Discretization for the output side:
                sentence_leve_latents_list.append(self.get_pose_latent(sample_skeletons))

            # if i>100:
            #     break



        object_to_save = {'sample_words_list':GPT_3_STR_list,
                          'GPT_3_Embedding_list': GPT_3_Embedding_list}
        pickle.dump(object_to_save, open(self.out_lmdb_dir + '_' + vid + '.gpt','wb'))
        print("Embedding Saved!")
        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                if self.sentence_level:
                    for words, poses, audio_raws, audio_mels, aux, sentence_leve_latents , GPT_3_Embedding in \
                            zip(sample_words_list, sample_skeletons_list,
                                sample_audio_list_raws, sample_audio_list_mels,
                                aux_info, sentence_leve_latents_list, GPT_3_Embedding_list):
                        poses = np.asarray(poses)
                        GPT_3_Embedding = np.array(GPT_3_Embedding)
                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [words, poses, audio_raws, audio_mels, aux, sentence_leve_latents, GPT_3_Embedding]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1
                else:
                    for words, poses, audio, aux in zip(sample_words_list, sample_skeletons_list,
                                                        sample_audio_list, aux_info):
                        poses = np.asarray(poses)

                        # save
                        k = '{:010}'.format(self.n_out_samples).encode('ascii')
                        v = [words, poses, audio, aux]
                        v = pyarrow.serialize(v).to_buffer()
                        txn.put(k, v)
                        self.n_out_samples += 1

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words


    def get_pose_latent(self, poses):

        # 1. Load models
        args, DAE, loss_fn, lang_model, out_dim = self.DAE_frame_level
        args, rnn, loss_fn, lang_model, out_dim = self.rnn_representation
        DAE.train(False)
        rnn.train(False)


        # poses, poses_mirror = process_bvh(bvh_file)

        mean = np.array(args.data_mean).squeeze()
        std = np.array(args.data_std).squeeze()
        std = np.clip(std, a_min=0.01, a_max=None)
        out_poses = (np.copy(poses) - mean) / std

        target = torch.from_numpy(out_poses)
        # target = torch.unsqueeze(target,2)
        target = target.to(device).float()
        reconstructed = []
        # for i in range(len(out_poses)):
        #     input = torch.unsqueeze(target[i],0)
        #     current_out = pose_decoder(input)
        #     reconstructed.append(current_out)

        if DAE.encoder == None:
            encoded = target
        else:
            encoded = DAE.encoder(target)
        # encoded = torch.squeeze(encoded, 2)
        # encoded = encoded.to('cpu')
        # encoded = encoded.detach().numpy()
        # all_frames_from_rnn = None

        result = np.zeros((len(encoded)//args.n_poses, rnn.decoder.n_layers, args.hidden_size))
        result_index = 0
        for i in range(0, len(encoded), args.n_poses):
            current_dict = dict()
            input_seq = encoded[i:i + args.n_poses]

            current_dict['original'] = poses[i: i + args.n_poses]
            current_dict['latent_linear'] = encoded[i: i + args.n_poses].detach().clone().to('cpu').numpy()

            input_pre_seq = encoded[i]
            output_seq = encoded[i:i + args.n_poses]

            input_seq = torch.unsqueeze(input_seq, 0)
            input_seq = input_seq.transpose(0, 1)
            use_drivitive = args.use_derivitive=='True'
            if use_drivitive:
                diff = [(input_seq[n, :] - input_seq[n - 1, :]) for n in range(1, input_seq.shape[0])]
                diff.insert(0, torch.zeros_like(input_seq[0, :]))
                input_seq = torch.cat((input_seq, torch.stack(diff)), dim=2)

            # output_seq = torch.unsqueeze(output_seq, 0)
            # output_seq = output_seq.transpose(0, 1)
            output_seq = input_seq.clone()




            reconstructed_rnn = torch.zeros(args.n_poses, output_seq.size(1), rnn.decoder.output_size) \
                .to(output_seq.device)
            # run words through encoder
            encoder_outputs, encoder_hidden = rnn.encoder(input_seq, None)

            if rnn.VAE:
                decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
                # [2, 128, 200]
                # print("decoder_hidden!!! org", decoder_hidden.shape)
                decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()  # [128, 2, 200]
                decoder_hidden = torch.reshape(decoder_hidden, (decoder_hidden.shape[0], -1))
                mean = rnn.VAE_fc_mean(decoder_hidden)
                logvar = rnn.VAE_fc_std(decoder_hidden)
                z = rnn.reparameterize(mean, logvar, train=False)
                z = rnn.VAE_fc_decoder(z)
                decoder_hidden = z.reshape(decoder_hidden.shape[0],
                                           rnn.decoder.n_layers, -1)
                decoder_hidden = decoder_hidden.transpose(1, 0).contiguous()
                # print("decoder_hidden!!! modified", decoder_hidden.shape)
                decoder_first_hidden = decoder_hidden
            else:
                decoder_hidden = encoder_hidden[:rnn.decoder.n_layers]  # use last hidden state from encoder
                # print("decoder_hidden!!! not VAE ", decoder_hidden.shape)

            current_dict['latent_rnn'] = torch.squeeze(decoder_hidden.detach().clone(), 1).to('cpu').numpy()
            result[result_index] = torch.squeeze(decoder_hidden.detach().clone(), 1).to('cpu').numpy()
            result_index = result_index + 1

        return result

    def GPT_3_caller(self, str_input, pickle_file_loaded):

        return 1

        if pickle_file_loaded != None:
            for i in range(len(pickle_file_loaded['sample_words_list'])):
                if str_input == pickle_file_loaded['sample_words_list'][i]:
                    print("Embedding found:", str_input)
                    return pickle_file_loaded['GPT_3_Embedding_list'][i]

        request = openai.Embedding.create(input=str_input, engine="text-similarity-ada-001")
        embedding = request['data'][0]['embedding']
        print("!!!!!!!!!!!!!!!!!!!Embedding requested:", str_input)
        return embedding

