import json
import os
import httpx
import asyncio
from tqdm.asyncio import tqdm
from src.repeated_utils import build_messages, write_json
import yaml
import argparse
import logging
import re
from datetime import datetime

from Lean4Verifier import Lean4Verifier




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

print(task)

input_file = os.path.join(config['root_dir'], config['version'], f'{task}.json')

# output_dir = os.path.join(config['root_dir'], 'solutions', f'{method_id}_{model_name}')
output_dir = os.path.join(config['root_dir'], config['solution_id'], method_id, model_name, task)


os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"{task}.json")

backup_file = os.path.join(output_dir, f"{task}-meta.json")



log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
price_log_path = os.path.join(output_dir, 'price', f'log_{task}_{log_timestamp}.txt')

os.makedirs(os.path.join(output_dir, 'price'), exist_ok=True)

price_file = config.get('price_file')

with open(price_file, 'r') as p:
    price_data = json.load(p)

log_path = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

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



# def extract_lean_code(text):
#     pattern = r"```lean4(.*?)```"
#     matches = re.findall(pattern, text, re.DOTALL)
#     return matches[0].strip() if matches else False


def remove_lean4_comments(code):
    """Remove both block comments (/- ... -/) and line comments (--) from Lean4 code."""
    # Remove block comments
    cleaned_code = code
    while (start := cleaned_code.find('/-')) != -1:
        if (end := cleaned_code.find('-/', start)) == -1:
            break
        cleaned_code = f"{cleaned_code[:start]}{cleaned_code[end + 2:]}"

    # Remove line comments and empty lines, including end-of-line comments
    cleaned_code = '\n'.join(
        line.split('--')[0].strip()  # Remove everything after '--' and strip whitespace
        for line in cleaned_code.splitlines()
        if line.split('--')[0].strip()  # Keep only non-empty lines
    )

    return cleaned_code.strip()


def extract_lean_code(text):
    pattern1 = r"```lean4(.*?)```"
    matches = re.findall(pattern1, text, re.DOTALL)

    if matches:
        solution = matches[-1].strip()

    pattern2 = r"```lean(.*?)```"
    matches = re.findall(pattern2, text, re.DOTALL)

    solution = matches[-1].strip() if matches else False
    
    if solution:
        cleaned_code = remove_lean4_comments(solution)
        if cleaned_code and cleaned_code != '':
            return cleaned_code
        
    else:
        return False
   


def judge_answer(code):
    v = Lean4Verifier(code = code, code_type = 'proof')
    return v.if_correct, v.error_message
  
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
        
        # print(result)
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


def build_complete_proof(context, lean4_code):
    
    proof_lines = lean4_code.split('\n')
    import_lines = [line for line in proof_lines if line.strip().startswith('import')]
    
    proof_lines = [line for line in proof_lines if not line.strip().startswith('import')]
    
    import_str = '\n'.join(import_lines)
    proof_str = '\n'.join(proof_lines)
    complete_proof = f"""
    {import_str}
    {context}
    
    {proof_str}
    """
    
    return complete_proof


def split_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

async def process_tasks(input_path):
    try:
        with open(input_path, 'r') as f:
            dataset = json.load(f)[config['start']: config['end']]
    except:
        with open(input_path, 'r') as f:
            dataset = json.load(f)
    
    # dataset = [d for d in dataset]
    
    write_json(backup_file, dataset)
    print(f'file backup at {backup_file}')
    
    
    
    # for d in dataset:
        # print(d)
        # d.pop('conjecture')
        # d.pop('response')
        # d.pop('score')
        # d.pop('time')
        
    current_dataset_ids = [d['id'] for d in dataset]

    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f3:
            results = json.load(f3)
            logging.info(f'Loaded results: {len(results)}')
            
            false_results = [r for r in results if r['score'] == False and r['time'] < 10]
            
            if len(false_results) != 0:
                time = false_results[0]['time'] + 1
            else:
                time = 11
            
                
            results  = [d for d in results if d['score'] == True or d['time'] ==10] 
            
            # results = [d for d in results if d['id'] in current_dataset_ids] 
            
            logging.info(f'Filtered results (valid data): {len(results)}')
            
            
            # Create a dictionary for quick lookup by id
            results_dict = {r['id']: r for r in false_results}

            # Replace dataset entries with corresponding results
            for i, d in enumerate(dataset):
                if d['id'] in results_dict:
                    
                    dataset[i] = results_dict[d['id']]
                else:
                    d['responses'] = []
                    d['solutions'] = []
                    score = False
            
                
                
    else:
        time = 1
        results = []
        logging.info('No existing results found')
        
        # initiate dataset
        for d in dataset:
            d['responses'] = []
            d['solutions'] = []
    
    print(f'time: {time} (count from 1)')
    
    processed_id = [d['id'] for d in results ]
    
    logging.info(f'Total dataset size: {len(dataset)}')
    logging.info(f'Already processed: {len(processed_id)}')
        
    print(len(dataset))
    print('processed: ' , len(processed_id))
    
    dataset = [d for d in dataset if d['id'] not in processed_id]
    
    print(f'to be processed: {len(dataset)}')
    
    logging.info(f'To be processed: {len(dataset)}')
    
    messages = build_messages(dataset, use_error=config['use_error'], use_system_prompt=config['use_system_prompt']) 

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
            
            print(response)
            
            dataset_index = start_index + j
            if dataset_index < len(dataset):
            # Skip empty responses
            # if dataset_index >= len(dataset) or not response or response[0] == '' or response[0].get('answer') == "'choices'":
            #     continue
                
                try:
                    lean4_code = extract_lean_code(response[0]['answer']) 
                except:
                    lean4_code = False
                    
                dataset[dataset_index]['responses'].append(response[0])
                
                if lean4_code:
                
                    complete_proof = build_complete_proof(dataset[dataset_index]['context'], lean4_code)
                    
                    score, error_message = judge_answer(code=complete_proof)
                    
                else: 
                    score = False
                    error_message = None
                
                dataset[dataset_index]['solutions'].append(
                    {
                        'code': lean4_code,
                        'error_msg': error_message,
                        'score': score
                    }
                )

                dataset[dataset_index]['score'] = score
                dataset[dataset_index]['time'] = len(dataset[dataset_index]['responses'])
                # dataset[dataset_index]['use_error'] = config['use_error']
                
                
        results.extend(dataset[start_index:start_index + len(response_message_batch)])

        save_json(results, output_file)
        
    pbar.close()  
    
    log_token_stats()

    
    correct_num = len([d for d in results if d.get('score', False)])
    all_num = len(results)
    accuracy = round(100 * correct_num / all_num, 2)
    
    task = config['task']
    print(f'api: task: {task}\t correct num: {correct_num}\t data num: {all_num}\t accuracy: {accuracy}\n')
    with open(log_path, "a") as log_file:
        
        log_file.write(f'api: task: {task}\t correct num: {correct_num}\t data num: {all_num}\t accuracy: {accuracy}\n')
    

async def main():
    
    with open(price_log_path, 'w') as log_file:
        log_file.write(f"Token statistics for model: {model_name}\n")
        log_file.write(f"Date: {datetime.now()}\n")
        log_file.write("=" * 50 + "\n\n")
        
        
    await process_tasks(input_file)
    
    
    
if __name__ == "__main__":
    
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        
    with open(price_log_path, 'a') as log_file:
        log_file.write(f"Date: {datetime.now()}\n")