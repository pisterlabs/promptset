from PIL import Image
import os

def crop(path, input, height, width, k, page, area):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            o = a.crop(area)
            o.save(os.path.join(path, "%s_IMG_%s.png" % (page, k)))
            k +=1

path = 'inception_dset_eg'

#width and height of imagenet dataset examples from openai microscope
height = 80
width = 80

#width and height of feature visualizations from openai microscope

k = 0
area = (0, 0, width, height)

for i in range(1, 50):
    page = "unit_%s" % i
    
    imgpath = "inception_dset_eg/channel_%s.png" % i
    crop(path, imgpath, height, width, k, page, area)