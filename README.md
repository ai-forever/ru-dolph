[[Paper]]() [[Хабр]]() [[Model Card]](https://huggingface.co/sberbank-ai/RuDOLPH-350M) [[Colab]](https://colab.research.google.com/drive/1gmTDA13u709OXiAeXWGm7sPixRhEJCga?usp=sharing) [[Kaggle]]()

Fine-Tuning example by [@Alex Wortega](https://github.com/AlexWortega) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12YRRzhl5cHER_U2F-buQxif8GlhMPWq3?usp=sharing)

## <img src="https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/rudolph.png?token=GHSAT0AAAAAABQH6MST7ZEGAF274DV33K7KYOYRSBQ" height="60"/> RuDOLPH 🦌🎄☃️

*One Hyper-Modal Transformer can be creative as DALL-E and smart as CLIP*

---

**Ru**ssian **D**iffusion **O**n **L**anguage **P**icture **H**yper-modality (RuDOLPH) is a fast 
and light text-image-text transformer (350M GPT-3) designed for a quick and easy fine-tuning setup 
for the solution of various tasks: from generating images by text description and image classification 
to visual question answering and more. This model demonstrates the power of Hyper-modality Transformers.

*(!!!) Hyper-modality means generalized multi-modal, e.g., model that consists of two multi-modal parts: text-2-image and image-2-text becomes text and image hyper-modality model*


![](./pics/scheme.png)

# Sparse Attention Mask
`row - col - row - [last] conv`

![](./pics/attention_masks.png)

# Models
+ [350M (RuDOLPH)](https://huggingface.co/sberbank-ai/RuDOLPH-350M)
+ 1.3B (In Progress)
+ 4B (In Progress)


![](./pics/high_res.png)

# Installing
```
pip install rudolph==0.0.1rc1
```

# Usage
### Init models
```python
from rudalle import get_tokenizer, get_vae
from rudalle.utils import seed_everything
from rudalle.image_prompts import ImagePrompts

from rudolph.model import get_rudolph_model
from rudolph.pipelines import zs_clf, generate_codebooks, self_reranking_by_image, self_reranking_by_text, show, generate_captions, generate_texts
from rudolph import utils

device = 'cuda'
model = get_rudolph_model('350M', fp16=True, device=device)
model.to(device);
tokenizer = get_tokenizer()
vae = get_vae(dwt=False).to(device)
```
### Text Generation
```python
generate_texts(
    tokenizer,
    model,
    template='красивый пейзаж ',
    top_k=32, top_p=0.6, texts_num=32, bs=32, seed=42
)[:8]

[{'text': 'красивый пейзаж с лесом и рекой. вид с воздуха на сельскую местность. пейзаж с лесом и рекой. вид на горы с беспилотника', 'ppl': 82.94},
 {'text': 'красивый пейзаж в стиле реализм, автор которой сергей владимирович дорофеев', 'ppl': 112.73},
 {'text': 'красивый пейзаж с рекой и озером - обои для рабочего стола, картинки, фото', 'ppl': 125.55},
 {'text': 'красивый пейзаж с рекой и мостом через реку в сумерках', 'ppl': 170.83},
 {'text': 'красивый пейзаж с горами в тумане - горы в тумане', 'ppl': 180.72},
 {'text': 'красивый пейзаж с лесом и лугом в сумерках', 'ppl': 185.84},
 {'text': 'красивый пейзаж с озером и лесом на заднем плане', 'ppl': 199.84},
 {'text': 'красивый пейзаж с видом на горы в таиланде', 'ppl': 219.86}]
```

### Setup for Fast Image Generation

```python
text = 'рисунок кота'
bs, images_num = 48, 48
top_k, top_p = 512, 0.9
with torch.no_grad():
    codebooks = generate_codebooks(text, tokenizer, model, top_k=top_k, images_num=images_num, top_p=top_p, bs=bs)
    ppl_text, ppl_image = self_reranking_by_text(text, codebooks, tokenizer, model, bs=bs)
    images = vae.decode(codebooks[ppl_text.argsort()[:4]])
images = torchvision.utils.make_grid(images, nrow=2)
img = torchvision.transforms.functional.to_pil_image(images)
img
```
![](./pics/pipelines/cat_drawing.png)

### Image Generation + Self Reranking
```python
text = 'красивый пейзаж с озером и лесом на заднем плане'
images_num = 256
seed_everything(42)
codebooks = []
for top_k, top_p, images_num in [
    (2048, 0.99, images_num),
    (1024, 0.99, images_num),
    (1024, 0.98, images_num),
]:
    codebooks.append(generate_codebooks(text, tokenizer, model, top_k=top_k, images_num=images_num, top_p=top_p, bs=32))

codebooks = torch.cat(codebooks)

ppl_text, ppl_image = self_reranking_by_text(text, codebooks, tokenizer, model, bs=32)
with torch.no_grad():
    images = vae.decode(codebooks[ppl_text.argsort()[:16]])

pil_images = utils.torch_tensors_to_pil_list(images)
show(pil_images, 8)
```
![](./pics/pipelines/lake.png)


```python
text = 'зимнее время года'

ppl_text, ppl_image = self_reranking_by_text(text, codebooks, tokenizer, model, bs=32)
with torch.no_grad():
    images = vae.decode(codebooks[ppl_text.argsort()[:16]])

pil_images = utils.torch_tensors_to_pil_list(images)
show(pil_images, 8)
```
![](./pics/pipelines/lake_winter.png)


```python
text = 'ночное время суток'

ppl_text, ppl_image = self_reranking_by_text(text, codebooks, tokenizer, model, bs=32)
with torch.no_grad():
    images = vae.decode(codebooks[ppl_text.argsort()[:16]])

pil_images = utils.torch_tensors_to_pil_list(images)
show(pil_images, 8)
```
![](./pics/pipelines/lake_night.png)


### Image Prompt (like Inpainting)
![](pics/pipelines/lake_image_prompt.png)
```python
text = 'лодка с алыми парусами'

images_num = 1024
bs = 32

borders = {'up': 6, 'left': 4, 'right': 6, 'down': 2}
image_prompts = ImagePrompts(pil_img, borders, vae, device, crop_first=True)

seed_everything(42)
codebooks = []
for top_k, top_p, images_num in [
    (1024, 0.99, images_num),
]:
    codebooks.append(
        generate_codebooks(text, tokenizer, model, top_k=top_k, images_num=images_num, top_p=top_p, bs=bs, image_prompts=image_prompts)
    )

codebooks = torch.cat(codebooks)

ppl_text, ppl_image = self_reranking_by_text(
    text,
    codebooks,
    tokenizer,
    model,
    bs=bs,
)
with torch.no_grad():
    images = vae.decode(codebooks[ppl_text.argsort()[:16]])

pil_images = utils.torch_tensors_to_pil_list(images)
show(pil_images, 8)
```
![](./pics/pipelines/lake_ship.png)

### Diffusion (TODO, see [Colab](https://colab.research.google.com/drive/1gmTDA13u709OXiAeXWGm7sPixRhEJCga?usp=sharing))

### Image Captioning + Self Reranking

```python
texts = generate_captions(pil_img, tokenizer, model, vae, template='на картинке ', top_k=8, captions_num=128, bs=32, top_p=0.6, seed=42)
ppl_text, ppl_image = self_reranking_by_image(texts, pil_img, tokenizer, model, vae, bs=32, seed=42)
for idx in ppl_image.argsort()[:8]:
    print(f'-{texts[idx]}')
```

![](./pics/pipelines/final_lake_ship.png)
```python
-на картинке я хочу увидеть как выглядит дом в горах
-на картинке нарисована лодка с каяком и лесом
-на картинке нарисован дом с бассейном
-на картинке – пейзаж – горы – одна из самых красивых мест на планете
-на картинке: в норвегии
-на картинке в горах
-на картинке я хочу нарисовать дом
-на картинке изображен домик на горе
```

![](./pics/pipelines/captioning_dog.png)
```python
-на картинке изображен рыжий пес. на фото изображен рыжий пес
-на картинке собака с длинным носом и длинным носом и короткой шерстью
-на картинке собака с длинными ушами и короткой шерстью
-на картинке изображена собака с большими глазами и длинным носом
-на картинке изображен белый медведь
-на картинке собака похожа на стаффорда и бультерьера. фото, на котором
-на картинке собака похожа на бигля и на собаку
-на картинке собака с длинными ушами и длинными ушами и
```

![](./pics/pipelines/captioning_street.png)
```python
-на картинке изображена улица с светофором
-на картинке изображен дом на участке ижс
-на картинке изображена дорога с двумя автомобилями
-на картинке изображен вид с воздуха на жилой район, который находится на улице и в районе жилого комплекса
-на картинке изображен вид на здание с окнами и окнами
-на картинке изображена дорога с светофором
-на картинке изображен дом напротив станции
-на картинке изображен жилой дом
```

![](./pics/pipelines/captioning_moto.png)
```python
-на картинке изображен мотоцикл иж юпитер
-на картинке изображена молодая женщина с каре на фоне деревянного дома
-на картинке изображён мотоцикл
-на картинке изображен велогонщик
-на картинке изображена мотокультиватор
-на картинке изображено здание
-на картинке изображена девушка с велосипедом
-на картинке изображен мопед
```

### Zero-Shot Image Classification using PPL
```python
import base64
import requests
from PIL import Image
from io import BytesIO

bs4_urls = requests.get('https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/pipelines/cats_vs_dogs_bs4.json').json()

f, ax = plt.subplots(2,4, figsize=(12,6))

for i, bs4_url in enumerate(bs4_urls):
    pil_img = Image.open(BytesIO(base64.b64decode(bs4_url)))
    
    classes = ['кошка', 'собака']
    preds = zs_clf(
        pil_img, 
        classes,
        model, 
        tokenizer,
        vae,
        template = 'на фото изображена', 
    )
    ax[i//4, i%4].imshow(pil_img)
    ax[i//4, i%4].set_title(preds['class'])
```
![](./pics/pipelines/zs_clf.png)

### Linear Probe (TODO, see [Colab](https://colab.research.google.com/drive/1gmTDA13u709OXiAeXWGm7sPixRhEJCga?usp=sharing))

# Authors: 

+ Alex Shonenkov: [Github](https://github.com/shonenkov), [Kaggle GM](https://www.kaggle.com/shonenkov)
+ Michael Konstantinov: [Mishin Learning](https://t.me/mishin_learning), [Transformer Community](https://transformer.community/)

<img src='https://habrastorage.org/webt/2w/5k/2r/2w5k2reyf6yqa4s7ywmmioaaieg.png' alt="Drawing" width="200" />  <img src='https://habrastorage.org/webt/eq/ft/g3/eqftg3_8l1b_fpimhiof7knytzk.png' alt="Drawing" width="200" />

# Citation

```
@article{shonenkov2022ruDolph,
  title         = {RuDOLPH: One Hyper-Modal Transformer can be creative as DALL-E and smart as CLIP},
  author        = {Alex Shonenkov and Michael Konstantinov},
  year          = {2022},
  eprint        = {...},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

```
@misc{github2022ruDolph,
  title         = {RuDOLPH: One Hyper-Modal Transformer can be creative as DALL-E and smart as CLIP},
  author        = {Alex Shonenkov and Michael Konstantinov},
  year          = {2022},
  howpublished  = {\url{https://github.com/sberbank-ai/ru-dolph}},
}
```

# Supported by

[<img src="https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/logo/sberai-logo.png" height="115"/>](https://github.com/sberbank-ai) \
[<img src="https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/logo/sberdevices-logo.png" height="40"/>](https://sberdevices.ru)

[<img src="https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/logo/sbercloud-logo.png" height="80"/>](https://sbercloud.ru/) \
[<img src="https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/logo/airi-logo.png" height="50"/>](https://airi.net)
