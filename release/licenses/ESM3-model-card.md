---
language:
- en
tags:
- biology
- esm
- protein
---

# Model Card for esm3-sm-open-v1

`esm3-sm-open-v1` is trained on 2.78 billion natural proteins. With synthetic data augmentation, this led to 3.15 billion protein sequences, 236 million protein structures, and 539 million proteins with function annotations, totaling 771 billion tokens.
`esm3-sm-open-v1` is a generative model capable of designing proteins conditioned on partial prompts of sequence, structure and function.

Safety is an important part of our model - data related to viruses has been removed from the training dataset, as well as some proteins belonging to organisms on the [USDA Select Agents and Toxins](https://www.selectagents.gov/sat/list.htm) list.
The function decoder has been filtered for potentially harmful keywords.

## Usage

Using `ESM3` requires [esm](https://github.com/evolutionaryscale/esm)

```
pip install esm
```

Please refer to the readme and notebooks in the [esm repository](https://github.com/evolutionaryscale/esm?tab=readme-ov-file#quickstart) for details on how to use the model.

## License

This repository is under a MIT [license](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md).
