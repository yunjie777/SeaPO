import argparse
import json
import os
from random import sample

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

prompt_template = """
Your task is to deliberately modify the provided response to introduce the specified error.

### Task:
Analyze the given question, original response, error type, and error description. Then, revise the response to intentionally include the specified error.

1. **Question**: {question}
2. **Original Response**: {response}
3. **Error Type**: {error_type}
4. **Error Description**: {error_description}

### Instructions:
- Modify the original response to clearly incorporate the specified error.
- Do not include any explanations, notes, or other text in your output.
- Output only the revised response.

### Revised Response:
"""

error_descriptions = {
    "correctness":
    "Mistakes related to factual accuracy or calculations, such as incorrect facts, reasoning errors, translation issues, or improper use of tools and formulas.",
    "logic":
    "Issues where the reasoning or argumentation is flawed, such as contradictions, unsupported conclusions, circular reasoning, or failure to follow a logical sequence in problem-solving or explanation.",
    "hallucination":
    "Instances where the model generates information that is completely fabricated or false, without basis in reality or relevant data, often resulting in the creation of nonexistent facts or misleading details.",
}


def clean_output(text: str) -> str:
    prefixes = [
        'Here is the revised response with the specified error:',
        '### Revised Response:'
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def main(args: argparse.Namespace):
    with open(args.file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    if args.debug:
        datas = sample(datas, 5)
        args.output_path = f'{os.path.basename(__file__)[:-3]}-debug.json'
    prompts = []
    llm = LLM(model=args.model_path,
              tensor_parallel_size=torch.cuda.device_count(),
              max_model_len=4096)
    for data in tqdm(datas, desc='Applying chat template...'):
        prompts.append(llm.get_tokenizer().apply_chat_template(
            [{
                'role':
                'user',
                'content':
                prompt_template.format(
                    question=data['prompt'],
                    response=data['chosen'][1]['content'],
                    error_type=args.error_type,
                    error_description=error_descriptions[args.error_type])
            }],
            tokenize=False,
            add_generation_prompt=True))
    result = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params)
    for output, data in zip(outputs, datas):
        result.append({'messages': data['messages'], 'label': True})
        result.append({
            'messages': [{
                'role': 'user',
                'content': data['prompt']
            }, {
                'role': 'assistant',
                'content': clean_output(output.outputs[0].text)
            }],
            'label':
            False
        })
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default='/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--error_type',
                        type=str,
                        choices=list(error_descriptions.keys()),
                        default='logic')
    parser.add_argument('--file_path',
                        type=str,
                        default='ultrafeedback.json')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
