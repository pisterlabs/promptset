import tensorflow as tf
import numpy as np
# from trans_dict import trans_dict_tf

""" to read the parameters of the gpt-2 pretrained model from tensorflow checkpoint
    and save them into npy files for mindspore to load.
    
    *this script is based on the gpt-2 model downloaded from openai.*
"""

# model_path = "G:/models/345M"
gpt2_checkpoint_path = "/home/tju/sdb3/oyy/124M/model.ckpt"
# gpt2_checkpoint_path = model_path + "/model.ckpt"
#model path and model name
init_vars = tf.train.list_variables(gpt2_checkpoint_path)
#load the model parameters into vars

save_param_num=0

for name, shape in init_vars:
    print('--name--:{}'.format(name))
    print('--shape--:{}'.format(shape))

    array = tf.train.load_variable(gpt2_checkpoint_path, name)
    print(array)

    # #print(name)
    # #By this you can understand the next step easily
    # name=name[6:].replace(r"/",".")
    # #skip 'model/' and change var names to avoid path mistake

    # if name not in trans_dict.keys():
    #     print(name + " is not in this model")
    # else:
    #     np.save(trans_dict[name] + ".npy", array)
    #     save_param_num = save_param_num + 1
    #save the parameters by 'npy'

print("finished!")
# print("save {num} parameters.".format(num=save_param_num))


