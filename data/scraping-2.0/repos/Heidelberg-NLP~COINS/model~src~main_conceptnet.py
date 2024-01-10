
import random

import torch
from transformers import GPT2Tokenizer

import src.train.conceptnet_train as train
import src.models.models as models
import src.models.model_knowledge_story as model_knowledge_story
import src.data.data as data
import utils.utils as utils
import src.train.utils as train_utils
import src.data.config as cfg
import numpy as np
#from src.data.utils import TextEncoder
from src.train.opt import OpenAIAdam
from src.train.opt_knowledge import Knowledge_Adam

def main(num):
    # Generate configuration files depending on experiment being run
    #utils.generate_config_files("conceptnet", num)

    # Loads the correct configuration file
    config_file = "config/conceptnet/config_{}.json".format(num)

    print(config_file)

    # Read config file to option
    config = cfg.read_config(cfg.load_config(config_file))
    opt, meta = cfg.get_parameters(config)

    # config.gpu_mode = torch.cuda.is_available()

    # Set the random seeds
    torch.manual_seed(opt.train.static.seed)
    random.seed(opt.train.static.seed)
    if config.gpu_mode:
        torch.cuda.manual_seed_all(opt.train.static.seed)

    # Load the data
    splits = ["train", "dev", "test"]

    opt.train.dynamic.epoch = 0

    print("Loading Data")
     
    # Initialize path to pre-set data loader
    x = "data/conceptnet/processed/generation/rel_language-trainsize_100-devversion_12-maxe1_200-maxe2_200.pickle"
    path = x.format(
        opt.exp, utils.make_name_string(opt.data))
    print(path)

    # Make data loader
    data_loader = data.make_data_loader(opt)
    loaded = data_loader.load_data(path)
    #print(data_loader.sequences["train"]["total"].size(0))
    data_loader.opt = opt
    data_loader.batch_size = opt.train.dynamic.bs

    print("Done.")
    print(data_loader)

    #text_encoder = TextEncoder(config.encoder_path, config.bpe_path)
    text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {"cls_token":"[CLS]", "unk_token":"[UNK]"}#, "mask": '["MASK"]',"separator": '["SEP"]', "start_of_sentence": '["SOS"]', "end_of_sentence": '["EOS"]'}
    text_encoder = GPT2Tokenizer.from_pretrained("gpt2", cls_token="[CLS]", unk_token="[UNK]", mask= '["MASK"]', separator='["SEP"]', start_of_sentence='["SOS"]', end_of_sentence='["EOS"]')
    text_encoder.add_special_tokens(special_tokens)
     
    #categories = data.conceptnet_data.conceptnet_relations
    
    special = [data.start_token, data.end_token]
    #special += ["<{}>".format(cat) for cat in categories]
 
    if loaded:
        text_encoder.encoder = data_loader.vocab_encoder
        text_encoder.decoder = data_loader.vocab_decoder
    else:
        for special_token in special:
            text_encoder.decoder[len(encoder)] = special_token
            text_encoder.encoder[special_token] = len(encoder)
        data_loader.make_tensors(text_encoder, special)
    
    # Set max size of different parts of relation
    context_size_i1 = data_loader.max_input1
    context_size_i2 = data_loader.max_input2
    context_size_i3 = data_loader.max_input3
    context_size_i4 = data_loader.max_input4
    context_size_o1 = data_loader.max_output1
    context_size_o2 = data_loader.max_output2
    context_size_o3 = data_loader.max_output3
    context_size_o4 = data_loader.max_output4
    
    #opt.data.maxr = context_size_r

    n_special = len(special)
    n_ctx = context_size_i1 + context_size_i2 + context_size_i3 + context_size_i4 + context_size_o1 + context_size_o2 + context_size_o3 + context_size_o4
    n_vocab = len(text_encoder.encoder) + n_ctx
    
    opt.net.vSize = n_vocab
    # Build Model
    print("Building Model")
    print(opt.net.init=="pt")
    model = models.make_model(
        opt, n_vocab, n_ctx, n_special)    
    model.resize_token_embeddings(len(text_encoder))
    
    model_knowledge = model_knowledge_story.make_model(
        opt, n_vocab, n_ctx, n_special)
    model_knowledge.resize_token_embeddings(len(text_encoder))
    
    print("Done.")

    print("Files will be logged at: {}".format(
        utils.make_name(opt, prefix="results/losses/",
                        is_dir=True, eval_=True)))

    data_loader.reset_offsets("train", keys=["total"])

    data.set_max_sizes(data_loader)

    # Push to GPU
    if config.gpu_mode:
        print("Pushing to GPU: {}".format(config.gpu_index))
        cfg.device = config.gpu_index
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        if config.multigpu:
            #print("!!! I am here !!!")
            model = models.multi_gpu(
                model, config.gpu_indices).cuda()
            #model.to(f'cuda:{model.device_ids[0]}')
            model_knowledge = model_knowledge_story.multi_gpu(
                model_knowledge, config.gpu_indices).cuda()
            #model_knowledge.to(f'cuda:{model.device_ids[1]}')
        else:
            model.cuda(cfg.device)
            model_knowledge.cuda(cfg.device)
        print("Done.")

    print("Training")

    optimizer_m = OpenAIAdam(model.parameters(),
                           lr=opt.train.dynamic.lr,
                           schedule=opt.train.static.lrsched,
                           warmup=opt.train.static.lrwarm,
                           t_total=meta.iterations,
                           b1=opt.train.static.b1,
                           b2=opt.train.static.b2,
                           e=opt.train.static.e,
                           l2=opt.train.static.l2,
                           vector_l2=opt.train.static.vl2,
                           max_grad_norm=opt.train.static.clip)

    optimizer_k = Knowledge_Adam(model_knowledge.parameters(),
                           lr=opt.train.dynamic.lr,
                           schedule=opt.train.static.lrsched,
                           warmup=opt.train.static.lrwarm,
                           t_total=meta.iterations,
                           b1=opt.train.static.b1,
                           b2=opt.train.static.b2,
                           e=opt.train.static.e,
                           l2=opt.train.static.l2,
                           vector_l2=opt.train.static.vl2,
                           max_grad_norm=opt.train.static.clip)
                    
    trainer = train.make_trainer(
        opt, meta, data_loader, model, model_knowledge, optimizer_m, optimizer_k)
    
    trainer.set_generator(opt, model, model_knowledge, data_loader)
    trainer.set_evaluator(opt, model, model_knowledge, data_loader)

    trainer.run()
