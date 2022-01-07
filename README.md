[[Paper]]() [[Хабр]]() [[Model Card]](https://huggingface.co/sberbank-ai/RuDOLPH-350M) [[Colab]]() [[Kaggle]]()

## <img src="https://raw.githubusercontent.com/shonenkov/ru-dolph/master/pics/rudolph.png?token=AHV2MCOWDUYEND527HLVOPDB3MLAK" height="60"/> RuDOLPH 🦌🎄☃️

*One Hyper-Modal Transformer can be creative as DALL-E and smart as CLIP*



**Ru**ssian **D**iffusion **O**n **L**anguage **P**icture **H**yper-modality (RuDOLPH) Transformer



# Sparse Attention Mask
`row - col - row - [last] conv`

![](./pics/attention_masks.png)

# Installing
```
pip install rudolph==0.0.1rc0
pip install rudalle==0.4.0
```

# Usage
**Init models**
```python
from rudalle import get_tokenizer, get_vae
from rudalle.utils import seed_everything
from rudalle.image_prompts import ImagePrompts

from rudolph.model import get_rudolph_model
from rudolph.pipelines import generate_codebooks, self_reranking_by_image, self_reranking_by_text, show, generate_captions, generate_texts

device = 'cuda'
model = get_rudolph_model('350M', fp16=True, device=device)
model.to(device);
tokenizer = get_tokenizer()
vae = get_vae(dwt=False).to(device)
```
**Text Generation** 
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
