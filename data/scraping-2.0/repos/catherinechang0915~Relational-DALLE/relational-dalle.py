import torch
from torchvision.utils import save_image
from torchvision import transforms
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer
from pathlib import Path
from einops import repeat
from model import RN
from config import TRAIN_CONFIG, IMAGE_SIZE, DALLE_PATH, RN_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_CONFIG['cuda'] = (device == 'cuda')

class RelationalDalle(torch.nn.Module):
    def __init__(self, dalle_path=DALLE_PATH, rn_path=RN_PATH):
        super().__init__()
        # load DALL-E and RN
        dalle_path = Path(dalle_path)
        rn_path = Path(rn_path)
        assert dalle_path.exists(), 'trained DALL-E must exist'
        assert rn_path.exists(), 'trained RelationalNetwork must exist'
        dalle_obj = torch.load(str(dalle_path))
        dalle_params, vae_params, weights, vae_class_name, version = dalle_obj.pop('hparams'), dalle_obj.pop('vae_params'), dalle_obj.pop('weights'), dalle_obj.pop('vae_class_name', None), dalle_obj.pop('version', None)
        vae = OpenAIDiscreteVAE()

        rn_obj = torch.load(str(rn_path))

        self.dalle = DALLE(vae = vae, **dalle_params).cuda()
        self.dalle.load_state_dict(weights)

        self.rn = RN(TRAIN_CONFIG).cuda()
        self.rn.load_state_dict(rn_obj)
        self.image_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))])


    def sentence_to_question(self, sentence):
        """
        Question encoding:
        0-5 correspond to o1 color
        6-7 correspond to o1 shape

        8-14 correspond to o2 color
        14-15 correspond to o2 shape
        [R, G, B, O, K, Y, circle, rectangle, R, G, B, O, K, Y, circle, rectangle]

        Sentence in format:
        A <o1 color> <o1 shape> is above <o2 color> <o2 shape>.
        """
        index_to_color = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
        words = sentence.strip().replace('.', '').split(' ')
        o1_color_idx = index_to_color.index(words[1])
        o1_shape_idx = 6 if words[2] == 'circle' else 7
        o2_color_idx = index_to_color.index(words[6]) + 8
        o2_shape_idx = 14 if words[7] == 'circle' else 15

        question = [0]*16
        question[o1_color_idx] = 1
        question[o1_shape_idx] = 1
        question[o2_color_idx] = 1
        question[o2_shape_idx] = 1
        return torch.tensor([question]).to(device)

    def generate_images(self, text, batch_size=64, output_dir_name='./outputs', num_images=3):
        text_tokens = tokenizer.tokenize([text], self.dalle.text_seq_len).cuda()
        text_tokens = repeat(text_tokens, '() n -> b n', b = num_images)

        outputs = []
        question = self.sentence_to_question(text)
        while len(outputs) != num_images:
            for text_chunk in text_tokens.split(batch_size):
                output_img = self.dalle.generate_images(text_chunk, filter_thres = 0.9)
                transformed_image = self.image_transform(output_img)
                # transformed image is (1, 3, 64, 64), question is (1, 16)
                rn_out = self.rn(transformed_image, question)
                is_correct = rn_out.data.max(1)[1].item()
                if is_correct:
                    outputs.append(output_img)

        Path(output_dir_name).mkdir(parents = True, exist_ok = True)
        file_name = f"{text.replace(' ', '_')[:(100)]}.png"
        file_path = f"{output_dir_name}/{file_name}"
        outputs = torch.cat(outputs)

        for i, image in enumerate(outputs):
            save_image(image, file_path, normalize=True)

    def forward(self, x):
        raise NotImplemented

if __name__ == '__main__':
    dalle = RelationalDalle()
    f = open("../dalle-test.txt", "r")
    lines = f.readlines()
    d = [
        ["../data/angela", lines[:250]],
        ["../data/vishaal", lines[250:500]],
        ["../data/adrian", lines[500:750]],
        ["../data/ruimeng", lines[750:]]
    ]
    for i in d:
        output_dir = i[0]
        lines = i[1]
        j = 0
        print("generating images in", output_dir)
        for l in lines:
            if j%10==0:
                print(j, "of 250")
            dalle.generate_images(l.strip(), output_dir_name=output_dir, num_images=1)