


from urllib import request


https://jaykmody.com/blog/gpt-from-scratch/

https://github.com/jaymody/picoGPT

https://news.ycombinator.com/item?id=34726115




cd
git clone https://github.com/jaymody/picoGPT
cd picoGPT

code requirements.txt

pip list|findstr   num

conda list|  findstr requests
conda list|  findstr tqdm
conda list|  findstr fire


#numpy==1.24.1 # used for the actual model code/weights
##numpy=1.24 # used for the actual model code/weights
#numpy==1.24.3 # used for the actual model code/weights
###regex==2017.4.5 # used by the bpe tokenizer
#####requests==2.27.1 # used to download gpt-2 files from openai
tqdm==4.64.0 # progress bar to keep your sanity
fire==0.5.0 # easy CLI creation

# used to load the gpt-2 weights from the open-ai tf checkpoint
# M1 Macbooks require tensorflow-macos
tensorflow==2.11.0; sys_platform != 'darwin' or platform_machine != 'arm64'
tensorflow-macos==2.11.0; sys_platform == 'darwin' and platform_machine == 'arm64'


###pip install   --dry-run   -r requirements.txt
##dry run
  ##Downloading tensorflow_intel-2.11.0-cp310-cp310-win_amd64.whl (266.3 MB)
















https://news.ycombinator.com/item?id=34726115

lvwarren 86 days ago | prev | next [–]

Make this change in utils.py:
  def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
       [...]
        #name = name.removeprefix("model/")
        name = name[len('model/'):]
and you're cool example will run in Google Colab under Python 3.8 otherwise the 3.9 Jupyter patching is a headache.




