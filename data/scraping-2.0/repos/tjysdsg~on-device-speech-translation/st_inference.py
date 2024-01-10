# Copied from ESPnet (espnet2/bin/st_inference.py) to support quantization

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import torch
from torch import nn
from typeguard import check_argument_types, check_return_type

from espnet2.st.espnet_model import ESPnetSTModel
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis
from espnet2.tasks.enh_s2t import EnhS2TTask
from espnet2.tasks.lm import LMTask
from espnet2.tasks.st import STTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

try:
    from transformers import AutoModelForSeq2SeqLM

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class Speech2Text(nn.Module):
    """Speech2Text class

    Examples:
        >>> import soundfile
        >>> speech2text = Speech2Text("st_config.yml", "st.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> speech2text(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            st_train_config: Union[Path, str] = None,
            st_model_file: Union[Path, str] = None,
            transducer_conf: dict = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            ngram_scorer: str = "full",
            ngram_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            src_lm_train_config: Union[Path, str] = None,
            src_lm_file: Union[Path, str] = None,
            src_ngram_scorer: str = "full",
            src_ngram_file: Union[Path, str] = None,
            src_token_type: str = None,
            src_bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            asr_maxlenratio: float = 0.0,
            asr_minlenratio: float = 0.0,
            batch_size: int = 1,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.0,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            asr_beam_size: int = 20,
            asr_lm_weight: float = 1.0,
            asr_ngram_weight: float = 0.9,
            asr_penalty: float = 0.0,
            asr_ctc_weight: float = 0.3,
            asr_nbest: int = 1,
            enh_s2t_task: bool = False,
            ctc_greedy: bool = False,
            hugging_face_decoder: bool = False,
            hugging_face_decoder_max_length: int = 256,

            # ================================
            quantized: bool = False,
            quantize_modules: List[str] = None,
            quantize_dtype: str = None,
            # ================================
    ):
        super().__init__()
        assert check_argument_types()

        task = STTask if not enh_s2t_task else EnhS2TTask

        # ================================
        self.quantized = quantized
        if quantized:
            quantize_modules = set([getattr(torch.nn, q) for q in quantize_modules])
            quantize_dtype = getattr(torch, quantize_dtype)
        # ================================

        # 1. Build ST model
        scorers = {}
        asr_scorers = {}
        st_model, st_train_args = task.build_model_from_file(
            st_train_config, st_model_file, device
        )
        if enh_s2t_task:
            st_model.inherite_attributes(
                inherite_s2t_attrs=[
                    "ctc",
                    "decoder",
                    "eos",
                    "joint_network",
                    "sos",
                    "token_list",
                    "use_transducer_decoder",
                ]
            )
        st_model.to(dtype=getattr(torch, dtype)).eval()

        # ================================
        if quantized:
            st_model = torch.quantization.quantize_dynamic(
                st_model, qconfig_spec=quantize_modules, dtype=quantize_dtype
            )
        # ================================

        if hasattr(st_model, "decoder"):
            decoder = st_model.decoder
        else:
            decoder = None
        token_list = st_model.token_list
        scorers.update(
            decoder=decoder,
            length_bonus=LengthBonus(len(token_list)),
        )

        if ctc_weight > 0:
            assert hasattr(st_model, "st_ctc")
            ctc = CTCPrefixScorer(ctc=st_model.st_ctc, eos=st_model.eos)
            scorers.update(ctc=ctc)

        src_token_list = st_model.src_token_list
        if st_model.use_multidecoder:
            asr_decoder = st_model.extra_asr_decoder
            asr_ctc = CTCPrefixScorer(ctc=st_model.ctc, eos=st_model.src_eos)
            asr_scorers.update(
                decoder=asr_decoder,
                ctc=asr_ctc,
                length_bonus=LengthBonus(len(src_token_list)),
            )
        else:
            asr_decoder = None

        # 2. Build Language model
        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            scorers["lm"] = lm.lm

        if src_lm_train_config is not None:
            src_lm, src_lm_train_args = LMTask.build_model_from_file(
                src_lm_train_config, src_lm_file, device
            )
            asr_scorers["lm"] = src_lm.lm

        # 3. Build ngram model
        if ngram_file is not None:
            if ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                ngram = NgramFullScorer(ngram_file, token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                ngram = NgramPartScorer(ngram_file, token_list)
        else:
            ngram = None
        scorers["ngram"] = ngram

        if src_ngram_file is not None:
            if src_ngram_scorer == "full":
                from espnet.nets.scorers.ngram import NgramFullScorer

                src_ngram = NgramFullScorer(src_ngram_file, src_token_list)
            else:
                from espnet.nets.scorers.ngram import NgramPartScorer

                src_ngram = NgramPartScorer(src_ngram_file, src_token_list)
        else:
            src_ngram = None
        asr_scorers["ngram"] = src_ngram

        # 4. Build BeamSearch object
        if st_model.st_use_transducer_decoder:
            beam_search_transducer = BeamSearchTransducer(
                decoder=st_model.decoder,
                joint_network=st_model.st_joint_network,
                beam_size=beam_size,
                lm=scorers["lm"] if "lm" in scorers else None,
                lm_weight=lm_weight,
                token_list=token_list,
                **transducer_conf,
            )

            beam_search = None
            hugging_face_model = None
            hugging_face_linear_in = None
        elif (
                decoder.__class__.__name__ == "HuggingFaceTransformersDecoder"
                and hugging_face_decoder
        ):
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available."
                    " Please install it via `pip install transformers`"
                    " or `cd /path/to/espnet/tools && . ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )

            hugging_face_model = AutoModelForSeq2SeqLM.from_pretrained(
                decoder.model_name_or_path
            )

            hugging_face_model.lm_head.load_state_dict(decoder.lm_head.state_dict())

            if hasattr(hugging_face_model, "model"):
                hugging_face_model.model.decoder.load_state_dict(
                    decoder.decoder.state_dict()
                )
                del hugging_face_model.model.encoder
            else:
                hugging_face_model.decoder.load_state_dict(decoder.decoder.state_dict())
                del hugging_face_model.encoder

            # del st_model.decoder.lm_head
            # del st_model.decoder.decoder

            hugging_face_linear_in = decoder.linear_in
            hugging_face_model.to(device=device).eval()

            # hacky way to use .score()
            st_model.decoder.hf_generate = hugging_face_model
            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )
            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=hugging_face_model.config.decoder_start_token_id,
                eos=hugging_face_model.config.eos_token_id,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key="full",
            )

            # beam_search = None
            beam_search_transducer = None

            # TODO(karita): make all scorers batchfied
            if batch_size == 1:
                non_batch = [
                    k
                    for k, v in beam_search.full_scorers.items()
                    if not isinstance(v, BatchScorerInterface)
                ]
                if len(non_batch) == 0:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
                else:
                    logging.warning(
                        f"As non-batch scorers {non_batch} are found, "
                        f"fall back to non-batch implementation."
                    )
            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
            logging.info(f"Decoding device={device}, dtype={dtype}")
        else:
            beam_search_transducer = None
            hugging_face_model = None
            hugging_face_linear_in = None

            weights = dict(
                decoder=1.0 - ctc_weight,
                ctc=ctc_weight,
                lm=lm_weight,
                ngram=ngram_weight,
                length_bonus=penalty,
            )
            beam_search = BeamSearch(
                beam_size=beam_size,
                weights=weights,
                scorers=scorers,
                sos=st_model.sos,
                eos=st_model.eos,
                vocab_size=len(token_list),
                token_list=token_list,
                pre_beam_score_key="full",
            )
            # TODO(karita): make all scorers batchfied
            if batch_size == 1:
                non_batch = [
                    k
                    for k, v in beam_search.full_scorers.items()
                    if not isinstance(v, BatchScorerInterface)
                ]
                if len(non_batch) == 0:
                    beam_search.__class__ = BatchBeamSearch
                    logging.info("BatchBeamSearch implementation is selected.")
                else:
                    logging.warning(
                        f"As non-batch scorers {non_batch} are found, "
                        f"fall back to non-batch implementation."
                    )
            beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
            for scorer in scorers.values():
                if isinstance(scorer, torch.nn.Module):
                    scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
            logging.info(f"Beam_search: {beam_search}")
            logging.info(f"Decoding device={device}, dtype={dtype}")

        asr_weights = dict(
            decoder=1.0 - asr_ctc_weight,
            ctc=asr_ctc_weight,
            lm=asr_lm_weight,
            ngram=asr_ngram_weight,
            length_bonus=asr_penalty,
        )
        asr_beam_search = BeamSearch(
            beam_size=asr_beam_size,
            weights=asr_weights,
            scorers=asr_scorers,
            sos=st_model.src_sos,
            eos=st_model.src_eos,
            vocab_size=len(src_token_list),
            token_list=src_token_list,
            pre_beam_score_key="full",
            return_hs=True,
        )
        # TODO(karita): make all scorers batchfied
        if batch_size == 1:
            non_batch = [
                k
                for k, v in asr_beam_search.full_scorers.items()
                if not isinstance(v, BatchScorerInterface)
            ]
            if len(non_batch) == 0:
                asr_beam_search.__class__ = BatchBeamSearch
                logging.info("BatchBeamSearch implementation is selected for ASR.")
            else:
                logging.warning(
                    f"As non-batch scorers {non_batch} are found, "
                    f"fall back to non-batch implementation."
                )
        asr_beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        for scorer in asr_scorers.values():
            if isinstance(scorer, torch.nn.Module):
                scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        logging.info(f"ASR Beam_search: {asr_beam_search}")
        logging.info(f"Decoding device={device}, dtype={dtype}")

        # 4. [Optional] Build Text converter: e.g. bpe-sym -> Text
        # compatibility for whisper tokenizer
        preprocessor_conf = getattr(st_train_args, "preprocessor_conf", {})
        whisper_language = preprocessor_conf.get("whisper_language", None)
        whisper_task = preprocessor_conf.get("whisper_task", None)
        if whisper_language:
            src_token_lang, token_lang = whisper_language
        else:
            src_token_lang, token_lang = None, None

        if token_type is None:
            token_type = st_train_args.token_type
        if bpemodel is None:
            bpemodel = st_train_args.bpemodel

        if token_type is None:
            tokenizer = None
        elif (
                token_type == "bpe"
                or token_type == "hugging_face"
                or "whisper" in token_type
        ):
            if bpemodel is not None:
                tokenizer = build_tokenizer(
                    token_type=token_type,
                    bpemodel=bpemodel,
                    whisper_language=token_lang,
                    whisper_task=whisper_task,
                )
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        if "whisper" in token_type:
            converter = OpenAIWhisperTokenIDConverter(
                model_type=bpemodel,
                language=token_lang or "en",
                task=whisper_task or "translate",
            )
            beam_search.set_hyp_primer(
                list(converter.tokenizer.sot_sequence_including_notimestamps)
            )
        else:
            converter = TokenIDConverter(token_list=token_list)
        logging.info(f"Text tokenizer: {tokenizer}")

        if src_token_type is None:
            src_token_type = st_train_args.src_token_type
        if src_bpemodel is None:
            src_bpemodel = st_train_args.src_bpemodel

        if src_token_type is None:
            src_tokenizer = None
        elif src_token_type == "bpe" or "whisper" in token_type:
            if src_bpemodel is not None:
                src_tokenizer = build_tokenizer(
                    token_type=src_token_type,
                    bpemodel=src_bpemodel,
                    whisper_language=src_token_lang,
                    whisper_task=whisper_task,
                )
            else:
                src_tokenizer = None
        else:
            src_tokenizer = build_tokenizer(token_type=src_token_type)
        if "whisper" in src_token_type:
            src_converter = OpenAIWhisperTokenIDConverter(
                model_type=src_bpemodel,
                language=src_token_lang or "en",
                task=whisper_task or "translate",
            )
            asr_beam_search.set_hyp_primer(
                list(src_converter.tokenizer.sot_sequence_including_notimestamps)
            )
        else:
            src_converter = TokenIDConverter(token_list=src_token_list)
        logging.info(f"Src Text tokenizer: {src_tokenizer}")

        self.st_model = st_model
        self.st_train_args = st_train_args
        self.converter = converter
        self.tokenizer = tokenizer
        self.src_converter = src_converter
        self.src_tokenizer = src_tokenizer
        self.beam_search = beam_search
        self.beam_search_transducer = beam_search_transducer
        self.hugging_face_model = hugging_face_model
        self.hugging_face_linear_in = hugging_face_linear_in
        self.hugging_face_beam_size = beam_size
        self.hugging_face_decoder_max_length = hugging_face_decoder_max_length
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        self.asr_beam_search = asr_beam_search
        self.asr_maxlenratio = asr_maxlenratio
        self.asr_minlenratio = asr_minlenratio
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.asr_nbest = asr_nbest
        self.ctc_greedy = ctc_greedy

    @torch.no_grad()
    def forward(
            self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[
        Tuple[Optional[str], List[str], List[int], Union[Hypothesis, TransHypothesis]]
    ]:
        """Inference

        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp

        """
        # FIXME: doesn't work in FX Graph Mode: assert check_argument_types()

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, _, asr_enc, _ = self.st_model.encode(**batch, return_int_enc=True)
        assert len(enc) == 1, len(enc)
        x = enc[0]

        # Multi-decoder ASR beam search
        if self.st_model.use_multidecoder:
            asr_nbest_hyps = self.asr_beam_search(
                x=asr_enc[0],
                maxlenratio=self.asr_maxlenratio,
                minlenratio=self.asr_minlenratio,
            )

            asr_results = []
            for hyp in asr_nbest_hyps:
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                if isinstance(hyp.hs, List):
                    asr_hs = torch.stack(hyp.hs)
                else:
                    asr_hs = hyp.hs

                src_token_int = hyp.yseq.tolist()
                src_token_int = list(
                    filter(lambda x: x != self.st_model.src_sos, src_token_int)
                )
                src_token_int = list(
                    filter(lambda x: x != self.st_model.src_eos, src_token_int)
                )

                # remove blank symbol id, which is assumed to be 0
                src_token_int = list(filter(lambda x: x != 0, src_token_int))

                # Change integer-ids to tokens
                src_token = self.src_converter.ids2tokens(src_token_int)

                if self.src_tokenizer is not None:
                    src_hyp_text = self.src_tokenizer.tokens2text(src_token)
                else:
                    src_hyp_text = None
                asr_results.append(
                    (src_hyp_text, src_token, src_token_int, hyp, asr_hs)
                )

            # Encode 1 best ASR result
            asr_hs = asr_results[0][-1].unsqueeze(0)
            asr_hs = to_device(asr_hs, device=self.device)
            asr_hs_lengths = asr_hs.new_full(
                [1], dtype=torch.long, fill_value=asr_hs.size(1)
            )
            md_enc, _, _ = self.st_model.md_encoder(asr_hs, asr_hs_lengths)
            x = md_enc[0]
            pre_x = enc[0]

        # c. Passed the encoder result and the beam search
        if self.ctc_greedy:
            from itertools import groupby

            lpz = self.st_model.st_ctc.argmax(enc)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != 0, collapsed_indices)]
            nbest_hyps = [
                {"score": 0.0, "yseq": [self.st_model.sos] + hyp + [self.st_model.eos]}
            ]
            nbest_hyps = [
                Hypothesis(
                    score=hyp["score"],
                    yseq=torch.tensor(hyp["yseq"]),
                )
                for hyp in nbest_hyps
            ]
        elif self.st_model.use_multidecoder and self.st_model.use_speech_attn:
            nbest_hyps = self.beam_search(
                x=x,
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
                pre_x=pre_x,
            )
        elif self.beam_search_transducer:
            logging.info("encoder output length: " + str(x.shape[0]))
            nbest_hyps = self.beam_search_transducer(x)

            best = nbest_hyps[0]
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(
                f"normalized log probability: {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                "best hypo: " + "".join(self.converter.ids2tokens(best.yseq[1:])) + "\n"
            )
        else:
            nbest_hyps = self.beam_search(
                x=x, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
            )
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.st_model.st_use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        if self.st_model.use_multidecoder:
            return (results, asr_results)
        assert check_return_type(results)
        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
        Returns:
            Speech2Text: Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)

    def export_encoder(self, speech: torch.Tensor, out_dir: str):
        self.eval()
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"]

        # b. Forward Encoder
        model: ESPnetSTModel = self.st_model

        # From self.st_model.encode(**batch, return_int_enc=True):
        feats, feats_lengths = model._extract_feats(speech, speech_lengths)
        if model.normalize is not None:
            feats, feats_lengths = model.normalize(feats, feats_lengths)

        out_path = os.path.join(out_dir, 'encoder.onnx')
        torch.onnx.export(
            model.encoder,
            (feats, feats_lengths),
            out_path,
            input_names=['feats', 'feats_lens'],
            output_names=[xs_pad, olens, None],  # the model's output names
            dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                          'output': {0: 'batch_size'}}
        )

        # Get encoder output so that it can be used for exporting decoder ONNX
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_onnx, encoder_out


class FxPTQFactory:
    """
    Factory utilities for post-training quantization using FX Graph Mode
    """

    def __init__(
            self,
            speech2text: Speech2Text,
            quantization: Literal['static', 'dynamic'] = 'static',
            # TODO:
            quantize_modules: List[str] = None,
            quantize_dtype: str = None,
    ):
        self.quantization = quantization
        if self.quantization == 'static':
            self.qconfig_mapping = get_default_qconfig_mapping('x86')
        elif self.quantization == 'dynamic':
            self.qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
        else:
            assert False, f'Invalid value for {quantization}'

        self.speech2text = speech2text
        assert not self.speech2text.quantized, \
            "Cannot use regular dynamic quantization implementation when using FX Graph Mode"

    def create_static_ptq(
            self,
            example_inputs,
    ) -> Speech2Text:
        q = quantize_fx.prepare_fx(
            self.speech2text,
            self.qconfig_mapping,
            example_inputs,
        )

        # Calibrate
        print(f'Calibrating PTQ using {len(example_inputs)} examples')
        with torch.no_grad():
            for data in example_inputs:
                q(data)

        q = quantize_fx.convert_fx(q)
        return q
