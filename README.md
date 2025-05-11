# Bringing Suzhou Numerals into the Digital Age: A Dataset and Recognition Study on Ancient Chinese Trade Records 🌟📜

## Overview 🌈
This project aims to digitize Suzhou numerals—an ancient Chinese numerical system used in historical trade records—by providing a curated dataset and a recognition model. Built upon [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch), we adapt and enhance it to recognize these unique brush-written characters, bridging the gap between historical archives and modern OCR technology. 🚀

### [**ALP 2025**](https://www.ancientnlp.com/alp2025/) Co-located with [**NAACL 2025**](https://2025.naacl.org/) 🎉

## Modifications ⚙️
- Added support for Suzhou numerals ✅
- Adjusted model parameters for our dataset 🔧
- Enhanced data preprocessing and augmentation ✨

## About the Project 📖
Suzhou numerals were once widely used in Chinese commerce from the Song Dynasty to the early 20th century. However, they are rarely supported in modern OCR systems, limiting access to historical trade documents. This project introduces:
- A dataset of 773 expert-annotated Suzhou numeral samples from Qing-era ledgers. 📊
- A statistical analysis of character usage in historical bookkeeping. 📈
- A baseline recognition model (CRNN with CTC loss) tailored for these low-resource, brush-written numerals. 🤖

Our goal is to enable researchers and developers to digitize and study ancient Chinese financial records. 🌍

## Dataset and Model in Action 🎥
### Dataset Source 📚
Our dataset comes from late Qing-era trade ledgers, such as this excerpt from the Hechang Firm in Nagasaki (*長崎和昌號*) (dated 1880s):

![Suzhou Numeral Ledger](assets/KM_48690-0002-u.jpg)

### Model Prediction Example 🔍
Here’s an example of how our CRNN model performs—and where it struggles:

<img src="assets/error_3.png" width="450" height="150" style="display: block; margin-left: auto; margin-right: auto;">

**What’s Happening**: The model misreads '〨〩' as '〧' because the strokes between the characters are nearly connected, a common challenge with brush-written numerals. 🤔

## Installation & Usage 🛠️

### Install Dependencies 📦
First, create virtual environment and install the required Python packages from `requirements.txt`:

```bash
# clone this repo
git clone https://github.com/morris0401/Suzhou-Numerals-Sequence-Recognition.git
cd Suzhou-Numerals-Sequence-Recognition

# create environment
conda create -n suzhou_numeral python=3.9
conda activate suzhou_numeral
pip install -r requirements.txt
```

### Running the Program ▶️
To train and evaluate the model, run `src/main.py` with the following command:

```bash
python src/main.py --image_dir ./data --batch_size 32 --num_epochs 20 --lr 1e-4 \
    --rotation_degree 20 --brightness 0 --contrast 0 --saturation 0 \
    --normalize_mean 0.5 --normalize_std 0.5
```

### Argument Options ⚙️
The script supports the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_dir` | `str` | `./data` | Path to the dataset directory. |
| `--batch_size` | `int` | `32` | Batch size for training. |
| `--num_epochs` | `int` | `20` | Number of training epochs. |
| `--lr` | `float` | `1e-4` | Learning rate. |
| `--rotation_degree` | `int` | `20` | Rotation degree for data augmentation. |
| `--brightness` | `float` | `0` | Brightness range for augmentation. |
| `--contrast` | `float` | `0` | Contrast adjustment for augmentation. |
| `--saturation` | `float` | `0` | Saturation adjustment for augmentation. |
| `--normalize_mean` | `float` | `0.5` | Normalization mean. |
| `--normalize_std` | `float` | `0.5` | Normalization standard deviation. |

You can modify these parameters as needed to adjust training and preprocessing behavior.

### Example Usage 💡
```bash
python src/main.py --image_dir ./custom_dataset --batch_size 64 --num_epochs 30
```

## License 📜
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation 📖
```
@inproceedings{wu-etal-2025-bringing,
    title = "Bringing Suzhou Numerals into the Digital Age: A Dataset and Recognition Study on {A}ncient {C}hinese Trade Records",
    author = "Wu, Ting-Lin  and
      Chen, Zih-Ching  and
      Chen, Chen-Yuan  and
      Chen, Pi-Jhong  and
      Wang, Li-Chiao",
    editor = "Anderson, Adam  and
      Gordin, Shai  and
      Li, Bin  and
      Liu, Yudong  and
      Passarotti, Marco C.  and
      Sprugnoli, Rachele",
    booktitle = "Proceedings of the Second Workshop on Ancient Language Processing",
    month = may,
    year = "2025",
    address = "The Albuquerque Convention Center, Laguna",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.alp-1.13/",
    pages = "105--111",
    ISBN = "979-8-89176-235-0",
    abstract = "Suzhou numerals, a specialized numerical no-tation system historically used in Chinese com-merce and accounting, played a pivotal role in financial transactions from the Song Dynasty to the early 20th century. Despite their his-torical significance, they remain largely absent from modern OCR benchmarks, limiting com-putational access to archival trade documents. This paper presents a curated dataset of 773 expert-annotated Suzhou numeral samples ex-tracted from late Qing-era trade ledgers. We provide a statistical analysis of character distri-butions, offering insights into their real-world usage in historical bookkeeping. Additionally, we evaluate baseline performance with hand-written text recognition (HTR) model, high-lighting the challenges of recognizing low-resource brush-written numerals. By introduc-ing this dataset and initial benchmark results, we aim to facilitate research in historical doc-umentation in ancient Chinese characters, ad-vancing the digitization of early Chinese finan-cial records. The dataset is publicly available at our huggingface hub, and our codebase can be accessed at our github repository."
}
```

## Original Repository 🔗
The original implementation can be found at:  
🔗 [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch)

## Acknowledgment 🙏
This project is based on [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch), which is licensed under the MIT License.
