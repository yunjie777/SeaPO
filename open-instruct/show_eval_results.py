import json
import os
import argparse
import pandas as pd

results_path = 'results'
dataset_keys = {
    'bbh': ['average_exact_match'],
    'codex_humaneval': ['pass@10'],
    'gsm': ['exact_match'],
    'MATH': ['accuracy'],
    'mmlu': ['average_acc'],
    'truthfulqa': ['MC1', 'MC2'],
}


def main(args: argparse.Namespace):
    datasets = [
        name for name in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, name))
    ]
    if args.models == 'all':
        models = [
            name
            for name in os.listdir(os.path.join(results_path, datasets[0]))
            if os.path.isdir(os.path.join(results_path, datasets[0], name))
        ]
    else:
        models = args.models.split(',')
    for dataset in datasets:
        print(f'{dataset}:')
        table = {'Metric': []}
        for model in models:
            table['Metric'].append(model)
            metrics_path = os.path.join(results_path, dataset, model,
                                        'metrics.json')
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            for key in dataset_keys[dataset]:
                if key not in table:
                    table[key] = []
                table[key].append(round(metrics[key] * 100, 2))
        df = pd.DataFrame(table)
        print(df.to_string(index=False) + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='Qwen2.5-7B-Instruct')
    args = parser.parse_args()
    main(args)
