# Fine_Tunning_of_LLAMA2_Model

# My HuggingFace Account : https://huggingface.co/pashd

# Data Set

- Original data: https://huggingface.co/datasets/timdettmers/openassistant-guanaco?row=0

- Reformat Data 1K sample: https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k?row=2

- Complete Reformat Data: https://huggingface.co/datasets/mlabonne/guanaco-llama2

- How to Prepare reformat data: https://colab.research.google.com/drive/1Ad7a9zMmkxuXTOh1Z7-rNSICA4dybpM2?usp=sharing

- How to push data to hub: https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html#:~:text=Upload%20your%20files,lines%2C%20text%2C%20and%20Parquet.





### How to fine tune Llama 2

- Free Google Colab offers a 15GB Graphics Card (Limited Resources --> Barely enough to store Llama 2–7b’s weights)

- Full fine-tuning is not possible here: we need parameter-efficient fine-tuning (PEFT) techniques like LoRA or QLoRA.

- To drastically reduce the VRAM usage, we must fine-tune the model in 4-bit precision, which is why we’ll use LoRA here.
