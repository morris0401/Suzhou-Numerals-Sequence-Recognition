# Suzhou-Numerals-Sequence-Recognition

This project is based on [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch).  
We have modified and extended the original code for Suzhou numeral recognition.

## Original Repository
The original implementation can be found at:  
🔗 [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch)

## Modifications
- Added support for Suzhou numerals
- Adjusted model parameters for our dataset
- Enhanced data preprocessing and augmentation

## Installation & Usage
### Prerequisites
Ensure you have **Python 3.9** installed. You can check your Python version using:

```bash
python --version
```

### Install Dependencies
First, install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Program
To train and evaluate the model, run `src/main.py` with the following command:

```bash
python src/main.py --image_dir ./data --batch_size 32 --num_epochs 20 --lr 1e-4 \
    --rotation_degree 20 --brightness 0 --contrast 0 --saturation 0 \
    --normalize_mean 0.5 --normalize_std 0.5
```

### Argument Options
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

### Example Usage
```bash
python src/main.py --batch_size 64 --num_epochs 30
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgment
This project is based on [GitYCC/crnn-pytorch](https://github.com/GitYCC/crnn-pytorch), which is licensed under the MIT License.
