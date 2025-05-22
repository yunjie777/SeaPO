import json

from datasets import load_dataset
from tqdm import tqdm

if __name__ == '__main__':
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                      split='train_prefs')
    result = []
    for data in tqdm(ds):
        result.append(data)
    with open('ultrafeedback.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
