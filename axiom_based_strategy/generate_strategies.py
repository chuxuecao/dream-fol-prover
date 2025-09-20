import json
import os
import httpx
import asyncio
from tqdm.asyncio import tqdm
import yaml
import argparse
import logging
import re
from datetime import datetime


import itertools
import random


def load_json(file):
    with open(file,'r', encoding="utf8") as load_f:
        data = json.load(load_f)
        return data
    
def write_json(file, dict):
    with open(file, "w", encoding="utf8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)



GENERATE_STRATEGY_SYSTEM_PROMPT: str = """
Task: Given axioms and a conjecture to prove, you are asked to generate a strategy.
You are also provided with specific axioms that you must apply in your strategy.

Instructions:
Your strategy should include a set of sub-propositions to be proved. For each sub-proposition, specify:

Sub-proposition: [Target to prove]
- Inference rules used: [Specific rules]
- Axioms used: [Relevant axioms]
- Previously proven lemmas used: [Results from previous sub-proofs] (if needed)

Output format:
[[STRATEGY]]
your strategy here
[[END]]
"""

GENERATE_STRATEGY_USER_PROMPT: str = """
[[AXIOMS]] {context}
[[CONJECTURE]] {conjecture}
[[AXIOMS TO FOCUS ON]] {selected_axioms}
[[STRATEGY]]
Please provide a strategy for the conjecture using the specified axioms. Do not provide any Lean 4 code.
"""

def build_messages(data, axiom_data, use_system_prompt = True):
 
    messages = []
    
    for d in data:
        
        for ax in axiom_data:
            if ax['id'] == d['id']:
                selected_axioms = ax['selected_axioms']
                break
        
        combinations = list(itertools.combinations(selected_axioms, 2))

        random.shuffle(combinations)

        # one_combination = combinations[0]
        
        # 初始化 combined_axioms
        if 'combined_axioms' not in d:
            d['combined_axioms'] = []

        one_combination = None

        # 遍历所有组合寻找不重复的组合
        for combination in combinations:
            if combination not in d['combined_axioms']:
                one_combination = combination
                break

        # 如果没有找到不重复的组合，使用第一个组合（如果有的话）
        if one_combination is None and combinations:
            one_combination = combinations[0]

        # 更新 combined_axioms
        if one_combination:
            d['combined_axioms'].append(one_combination)
        
    
        if use_system_prompt:
            messages.append([
                        {'role': 'system', 'content': GENERATE_STRATEGY_SYSTEM_PROMPT},
                        {'role': 'user', 'content': GENERATE_STRATEGY_USER_PROMPT.format(
                            context=d['context'],
                            conjecture = d['conjecture'],
                            selected_axioms=one_combination)}
            ])
            
        else:
            messages.append([
                    {
                        'role': 'user', 
                        'content': f"{GENERATE_STRATEGY_SYSTEM_PROMPT}\n{GENERATE_STRATEGY_USER_PROMPT.format(context=d['context'],conjecture = d['conjecture'], selected_axioms=one_combination)}"
                    }
                ]
            )
        
    # for m in messages:
    #     print(m, end='-----------------\n')
    
    print(messages[0])
    # print(len(messages))

    return messages



parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)


task = config['task']
method_id = config['method_id']

model_name = config['model_id']

solution_id = config['solution_id']

print(task)

root_dir = os.path.join(config['root_dir'], solution_id, method_id, model_name, task)

axiom_path = os.path.join(root_dir, "first-level-axioms.json")

output_file = os.path.join(root_dir, f"{task}.json")
input_file = output_file

log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
price_log_path = os.path.join(root_dir, 'price', f'log_{task}_{log_timestamp}.txt')

os.makedirs(os.path.join(root_dir, 'price'), exist_ok=True)

price_file = config.get('price_file')


with open(price_file, 'r') as p:
    price_data = json.load(p)

log_path = os.path.join(root_dir, 'log.txt')

if not os.path.exists(root_dir):
    os.makedirs(root_dir)
    

price = price_data[model_name]

print(model_name)


headers = {
    'Authorization': config['api_key'],
    'Content-Type': 'application/json',
}

CONCURRENCY = config.get('concurrency')

semaphore = asyncio.Semaphore(CONCURRENCY)


token_stats = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'prompt_price': 0,
    'completion_price': 0,
    'total_price': 0,
    'requests_count': 0
}

def log_token_stats():
    with open(price_log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("SUMMARY STATISTICS\n")
        log_file.write(f"Total requests: {token_stats['requests_count']}\n")
        log_file.write(f"Total prompt tokens: {token_stats['total_prompt_tokens']}\n")
        log_file.write(f"Total completion tokens: {token_stats['total_completion_tokens']}\n")
        log_file.write(f"Total tokens: {token_stats['total_tokens']}\n")
        log_file.write(f"Total prompt price: {token_stats['prompt_price']}\n")
        log_file.write(f"Total completion price: {token_stats['completion_price']}\n")
        log_file.write(f"Total price: {token_stats['total_price']}\n")
        
        if token_stats['requests_count'] > 0:
            avg_prompt = token_stats['total_prompt_tokens'] / token_stats['requests_count']
            avg_completion = token_stats['total_completion_tokens'] / token_stats['requests_count']
            avg_total = token_stats['total_tokens'] / token_stats['requests_count']
            log_file.write(f"Average prompt tokens per request: {avg_prompt}\n")
            log_file.write(f"Average completion tokens per request: {avg_completion}\n")
            log_file.write(f"Average total tokens per request: {avg_total}\n")
            log_file.write(f"Average prompt price per request: {token_stats['prompt_price'] / token_stats['requests_count']}\n")
            log_file.write(f"Average completion price per request: {token_stats['completion_price'] / token_stats['requests_count']}\n")
            log_file.write(f"Average total price per request: {token_stats['total_price'] / token_stats['requests_count']}\n")







def extract_strategy(text):
    match = re.search(r'\[\[STRATEGY\]\](.*?)\[\[END\]\]', text, re.DOTALL)
    if match:
        return match.group(1).strip()  
    return None  

   

  
async def get_batch_response_with_retry(messages, n=1, max_retries=3):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await get_batch_response(messages, n)
        except Exception as e:
            last_exception = e
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.error(f"Failed after {max_retries} attempts")
                raise last_exception
            

from openai import OpenAI

async def send_request_deepseek(client, json_data, n):
    prompt_answers = []
    for _ in range(n):
        client = OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        
        result = client.chat.completions.create(
            messages=json_data['messages'],
            model=json_data['model'],
            temperature=json_data['temperature'],
            max_tokens=32000,   
        )
        
        ans = result.choices[0].message.content
        
        usage = result.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # 其他统计逻辑保持不变
        prompt_answers.append({
            'answer': ans,
            'prompt_tokens': prompt_tokens, 
            'completion_tokens': completion_tokens
        })
        
    return prompt_answers


async def get_batch_response(messages, n=1):
    all_answers = []
    async with semaphore:
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0), limits=limits) as client: 
            tasks = []
            for message in messages:
                json_data = {
                    'model': model_name,
                    'messages': message,
                    'temperature': config.get('temperature', 1.0) + config.get('add_temperature', 0.0),
                    'stream': False,
                }
                # task = asyncio.create_task(send_request(client, json_data, n))
                
                if 'DeepSeek-Prover-V2-7B' not in model_name:
                    task = asyncio.create_task(send_request(client, json_data, n))
                else:
                    task = asyncio.create_task(send_request_deepseek(client, json_data, n))
                    
                    
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Error in batch: {str(result)}")
                    all_answers.append([str(result)])
                else:
                    all_answers.append(result)
    return all_answers

async def send_request(client, json_data, n):
    prompt_answers = []
    for _ in range(n):
        # try:
        response = await client.post(config['base_url'], headers=headers, json=json_data)
        # print(response)
        result = response.json()
        # print(result)
        ans = result['choices'][0]['message']['content']
        
        prompt_tokens = result['usage']['prompt_tokens']
        completion_tokens = result['usage']['completion_tokens']
        total_tokens = result['usage']['total_tokens']
        
        prompt_price = prompt_tokens * price['prompt']
        completion_price = completion_tokens * price['completion']
        all_price = prompt_price + completion_price
        
        token_stats['total_prompt_tokens'] += prompt_tokens
        token_stats['total_completion_tokens'] += completion_tokens
        token_stats['total_tokens'] += total_tokens
        token_stats['prompt_price'] += prompt_price
        token_stats['completion_price'] += completion_price
        token_stats['total_price'] += all_price
        token_stats['requests_count'] += 1
        
        with open(price_log_path, 'a') as log_file:
            log_file.write(f"Request ID: {token_stats['requests_count']}\n")
            log_file.write(f"  Prompt tokens: {prompt_tokens}\n")
            log_file.write(f"  Completion tokens: {completion_tokens}\n")
            log_file.write(f"  Total tokens: {total_tokens}\n")
            log_file.write(f"  Prompt price: {prompt_price}\n")
            log_file.write(f"  Completion price: {completion_price}\n")
            log_file.write(f"  Total price: {all_price}\n")
            log_file.write("-" * 40 + "\n")
        
        # prompt_answers.append(ans)
        
        prompt_answers.append({
            'answer': ans,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        })
        
        
    return prompt_answers

def save_json(data, file_name):
    json_data = json.dumps(data, indent=2, ensure_ascii=False)
    with open(file_name, 'w') as file:
        file.write(json_data)
    file.close()

def split_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

async def process_tasks(input_path, axiom_path):
    try:
        with open(input_path, 'r') as f:
            dataset = json.load(f)[config['start']: config['end']]
    except:
        with open(input_path, 'r') as f:
            dataset = json.load(f)
    
    with open(axiom_path, 'r') as f2:
        axiom_data = json.load(f2)
        

    if os.path.exists(output_file):
        with open(output_file, 'r') as f3:
            results = json.load(f3)
            logging.info(f'Loaded results: {len(results)}')
            
            # false_results = [r for r in results if r['score'] == False and r['time'] < 10]
 
            
            # results  = [d for d in results if d['score'] == True or d['time'] ==10] 
            
            results = [d for d in results if d['score'] or d['time'] == 10]
            
            to_be_processed_num = len(dataset) - len(results)
             
            logging.info(f'Filtered results (valid data): {len(results)}')
            
                
                
    else:
        results = []
        logging.info('No existing results found')
 
    processed_id = [d['id'] for d in results ]
    
    logging.info(f'Total dataset size: {len(dataset)}')
    logging.info(f'Already processed: {len(processed_id)}')
        
    print(len(dataset))
    print('processed: ' , len(processed_id))
    
    dataset = [d for d in dataset if d['id'] not in processed_id]
    
    for d in dataset:
        if not 'strategy-responses' in d:
            d['strategy-responses'] = []
    
    print(f'to be processed: {len(dataset)}')
    
    logging.info(f'To be processed: {len(dataset)}')
    
    
    messages = build_messages(dataset, axiom_data, use_system_prompt=config['use_system_prompt']) 

    tasks = []
    total_batches = len(list(split_into_batches(messages, config['batch_size'])))
    pbar = tqdm(total=total_batches, desc="Processing batches")

    for batch in split_into_batches(messages, config['batch_size']):
        task = asyncio.create_task(get_batch_response_with_retry(batch, n=config['n']))
        tasks.append(task)
    
    response_message_batches = await asyncio.gather(*tasks)

    for i, response_message_batch in enumerate(response_message_batches):
        pbar.update(1)  
        start_index = i * config['batch_size']
        for j, response in enumerate(response_message_batch):
            dataset_index = start_index + j
            if dataset_index < len(dataset) and response[0]['answer'] != "'choices'":
                
                ### process output
    
                try:
                    dataset[dataset_index]['strategy-responses'].append(response[0])
                    
                    dataset[dataset_index]['solutions'][-1]['strategy'] = extract_strategy(response[0]['answer'])
                    
                except:
                    
                    dataset[dataset_index]['strategy-responses'].append(False)
                    dataset[dataset_index]['solutions'][-1]['strategy'] = False
                
                
        results.extend(dataset[start_index:start_index + len(response_message_batch)])

        save_json(results, output_file)
            
    pbar.close()  
    
    log_token_stats()

    
    # pass_num = len([d for d in results if d.get('strategy-responses')[-1]])
    
    pass_num = 0
    for d in results:
        if d.get('strategy-responses', None):
            if d['strategy-responses'][-1]:
                pass_num += 1
    
    
    
    task = config['task']
    print(f'api: task: {task}\t pass num: {pass_num}\t data num: {to_be_processed_num}')
  

async def main():
    
    with open(price_log_path, 'w') as log_file:
        log_file.write(f"Token statistics for model: {model_name}\n")
        log_file.write(f"Date: {datetime.now()}\n")
        log_file.write("=" * 50 + "\n\n")
        
        
    await process_tasks(input_file, axiom_path)
    
    
    
if __name__ == "__main__":
    
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        
    with open(price_log_path, 'a') as log_file:
        log_file.write(f"Date: {datetime.now()}\n")