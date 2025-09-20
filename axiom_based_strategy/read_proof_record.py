import json
import os

def load_json(file):
    with open(file,'r', encoding="utf8") as load_f:
        data = json.load(load_f)
        return data
    
def write_json(file, dict):
    with open(file, "w", encoding="utf8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)


# task = 'SET001'
# task = 'GRP005'
task = 'GEO006'

input_dir = f'/{task}'
output_dir = f'/{task}'


os.makedirs(output_dir, exist_ok=True)

load_time = 7

for file in os.listdir(input_dir):
    if file.endswith('.json'):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        
        if os.path.isfile(output_file):
            print('Warning!!!!! File already exist')
            
        else:
            data = load_json(input_file)
            
            new_data = []
            
            for item in data:
                if item['score'] and item['time'] <=load_time:
                    new_data.append(item)
                else:  
                    item['responses'] = item['responses'][0:load_time]
                    item['solutions'] = item['solutions'][0:load_time]
                    item['score'] = False
                    item['time'] = load_time
                    
                    item["sub-prop-responses"] = item["sub-prop-responses"][0:load_time]
                    
                    item["analysis-responses"] = item["analysis-responses"][0:load_time]
                    
                    new_data.append(item)
                
            write_json(output_file, new_data)
