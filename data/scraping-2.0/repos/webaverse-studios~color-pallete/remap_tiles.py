from PIL import Image
import openai
import ast

from recolor import transfer, util, palette

def remap_colors(rgb_channels, rgb_values, num_colors):
    image_lab = transfer.rgb2lab(rgb_channels)
    mypalette = palette.build_palette(image_lab, num_colors)
    new_color_image = Image.new(mode="RGB", size=(1,1))
    new_color_image.putpixel( (0, 0), rgb_values )
    new_lab_image = transfer.rgb2lab(new_color_image)
    lab_color =new_lab_image.getpixel((0,0))
    mypalette.append((0,128,128))
    new_palette = mypalette.copy()
    new_palette[0] = lab_color
    image_lab_m = transfer.image_transfer(image_lab, mypalette, new_palette, sample_level=10, luminance_flag=0)
    return util.lab2rgb(image_lab_m)

def getRGB(object, adjective):
    outtext = openai.Completion.create(
        model="davinci",
        prompt="* the main color of grass in a chocolate world is light brown.\n* the main color of rocks in a lemon world is yellow.\n* the main color of "+object+" in a "+adjective+ " world is ",
        max_tokens=256,
        temperature=0,
        stop=['\n','.']
        )
    response = outtext.choices[0].text
    print('generated color')
    print(response)
    outtext = openai.Completion.create(
        model="davinci",
        prompt="* color: red rgb: (255,0,0)\n* color: blue rgb: (0,0,255)\n* color: "+response+" rgb: ",
        max_tokens=256,
        temperature=0,
        stop=['\n','*']
        )
    response = outtext.choices[0].text
    return response


if __name__ == "__main__":

    object = 'sand'
    adjective = 'emerald'
    rgb_image = Image.open('resized.png')
    openai.api_key = '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'
    generated_color = getRGB(object, adjective)
    remapped = remap_colors(rgb_image, ast.literal_eval(generated_color.strip()), 2)
    remapped.show()