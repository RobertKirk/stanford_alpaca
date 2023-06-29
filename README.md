
<p align="center" width="100%">
<img src="assets/logo.png" alt="Stanford-Alpaca" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

# Fork of: Stanford Alpaca

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a fork of the [Stanford Alpaca repo](https://github.com/tatsu-lab/stanford_alpaca) with adjustments and additional to enable generation of the in-distribution test dataset and the Sequential Instructions dataset used in [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452). The generated datasets can be found here:
- [Sequential Instructions](https://huggingface.co/datasets/UCL-DARK/sequential-instructions)
- [In-Distribution Test](https://huggingface.co/datasets/UCL-DARK/alpaca-farm-id-test)

To reproduce the generation of the Sequential Instructions dataset, follow the instructions in the [Data Generation Process](#data-generation-process) section below, but use `python -m generate_instruction_sequential generate_instruction_following_data`. This script also has the option of automatically uploading the generated dataset to huggingface using the `--save_to_hf=<organisation>/<dataset_name>` argument.

For the in-distribution test dataset, follow the instructions in the [Data Generation Process](#data-generation-process) section below as-is.

Otherwise, we recommend using the original repository, which has detailed instructions on the rest of the code.

# Data Generation Process

Running the code

- Set environment variables `OPENAI_API_KEY` to your OpenAI API key.
- Install the dependencies with `pip install -r requirements`.txt.
- Run `python -m generate_instruction generate_instruction_following_data` to generate the data.
- Optionally pass `--save_to_hf=<organisation>/<dataset_name>` to automatically upload the generated dataset to huggingface.

### Citation

Please cite the original repo if you use the data or code in this repo, as well as our paper:

```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```

```
@misc{kirkUnderstandingEffectsRLHF2023,
  title = {Understanding the {{Effects}} of {{RLHF}} on {{LLM Generalisation}} and {{Diversity}}},
  author = {Kirk, Robert and Mediratta, Ishita and Nalmpantis, Christoforos and Luketina, Jelena and Hambro, Eric and Grefenstette, Edward and Raileanu, Roberta},
  year = {2023},
  month = oct,
  number = {arXiv:2310.06452},
  eprint = {2310.06452},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2310.06452},
  urldate = {2023-10-26},
  archiveprefix = {arxiv},
}
```

Naturally, you should also cite the original [LLaMA paper](https://arxiv.org/abs/2302.13971) and the [Self-Instruct paper](https://arxiv.org/abs/2212.10560) if you use the code or data from this repo.

### Acknowledgements

We thank the original Alpaca authors for releasing their code.
