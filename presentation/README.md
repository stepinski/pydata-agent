when running code you have to create conda env with
todo
than patch the file with
/Users/unman/miniforge3/envs/ocr-pipeline/lib/python3.10/site-packages/craft_text_detector/models/basenet/vgg16_bn.py
commenting out lines
from torchvision.models.vgg import model_urls
and
model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
