import torch
from comet.interactive import functions as interactive
import comet.train.atomic_train as train
from comet.train.opt import OpenAIAdam
import comet.data.config as cfg

num_calibration_batches = 10

opt, state_dict = interactive.load_model_file("models/6.25e-05_adam_64_20500.pickle")

data_loader, text_encoder = interactive.load_data("atomic", opt)

n_ctx = data_loader.max_event + data_loader.max_effect
n_vocab = len(text_encoder.encoder) + n_ctx
model = interactive.make_model(opt, n_vocab, n_ctx, state_dict).to('cpu')
model.eval()


# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model.qconfig = torch.quantization.default_qconfig
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
config_file = "config/atomic/config_{}.json".format(0)
config = cfg.read_config(cfg.load_config(config_file))
opt, meta = cfg.get_parameters(config)

# Calibrate with the training set
model.eval()
optimizer = OpenAIAdam(model.parameters(),
                           lr=opt.train.dynamic.lr,
                           schedule=opt.train.static.lrsched,
                           warmup=opt.train.static.lrwarm,
                           t_total=100,
                           b1=opt.train.static.b1,
                           b2=opt.train.static.b2,
                           e=opt.train.static.e,
                           l2=opt.train.static.l2,
                           vector_l2=opt.train.static.vl2,
                           max_grad_norm=opt.train.static.clip)

trainer = train.make_trainer(
    opt, meta, data_loader, model, optimizer)
trainer.set_evaluator(opt, model, data_loader)
trainer.opt.train.dynamic.epoch = 0
trainer.run_evaluation_cycle()

print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

trainer.save_model()


#top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
#print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
