import re
import torch
import hf_olmo
import pandas as pd
import argparse
import json
import random
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tqdm.pandas()


def send_query_to_llm_chat(query, model, tokenizer, num_return_sequences):
    # feed query to model, and then generate response
    # if "gemma" in model.config.model_type or "llama" in model.config.model_type:
    #     query = tokenizer.apply_chat_template(messages, tokenize=False)
    print(model.device)
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(model.config, "max_sequence_length"):
        max_length = model.config.max_sequence_length
    else:
        max_length = 2048
    print("input length: ", input_length, max_length - input_length - 10)

    if model.config.model_type == "llama":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]

    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, do_sample=True, top_p=0.9,
                                      repetition_penalty=1.25, temperature=0.8, max_new_tokens=max_length - input_length - 10,
                                      eos_token_id=terminators, num_return_sequences=num_return_sequences)
        generate_response = tokenizer.batch_decode(generate_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    print(generate_response)
    return generate_response

def extract_answer_chat_yes_no(answers):
    extracted = []
    if isinstance(answers, str):
        answers = eval(answers)
    for answer in answers:
        answer = answer.replace('*', '')
        words = re.findall(r'\w+', answer.lower())

        yes_exist = False
        no_exist = False
        for word in words:
            if word == "yes":
                yes_exist = True
            elif word == "no":
                no_exist = True
        #falcon extraction
        # for word in words:
        #     if word == "yes":
        #         yes_exist = True
        #         break
        #     elif word == "no":
        #         no_exist = True
        #         break
        if yes_exist:
            extracted.append('yes')
        elif no_exist:
            extracted.append('no')
        else:
            extracted.append('fail')

    return extracted

def majority_vote(answers):
    if isinstance(answers, str):
        answers = eval(answers)
    yes_count = 0
    no_count = 0
    for answer in answers:
        if answer == "yes":
            yes_count += 1
        elif answer == "no":
            no_count += 1
    if yes_count >= no_count:
        return "yes"
    else:
        return "no"

demos_1 = [
    # {"role": "system", "content": "You are an expert in causal discovery, consistently and accurately identifying the correct causal relationships between two events.",
    # },
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'it is raining' and 'carrying an umbrella'. "
        "Does 'it is raining' cause 'carrying an umbrella'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The presence of rain often results in individuals using umbrellas. "
        "2. Rationale: People use umbrellas to protect themselves from getting wet during rain. "
        "3. Conclusion: Rain acts as a cause for the use of an umbrella. The final answer is Yes. "
        "Final Answer: Yes"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'it is raining' and 'carrying an umbrella'. "
        "Does 'carrying an umbrella' causes 'it is raining'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no. "},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The act of opening an umbrella does not influence weather patterns or conditions. "
        "2. Rationale: Rain is a meteorological event determined by atmospheric conditions. The use of an umbrella does not have the capability to influence weather conditions. "
        "3. Conclusion: There is no causal relationship where the umbrella causes rain. The answer is No. "
        "Final Answer: No "},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'Eating food' and 'feeling Hunger'. "
        "Does 'Eating food' causes 'feeling Hunger'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no. "},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Eating involves consuming food, which typically reduces the feeling of hunger. "
        "2. Rationale: The act of eating is intended to satiate hunger, not induce it. "
        "3. Conclusion: Eating does not cause hunger. Instead, it alleviates it. The answer is No. "
        "Final Answer: No"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'Eating food' and 'feeling Hunger'. "
        "Does 'Feeling hunger' causes 'Eating food'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: The sensation of hunger typically motivates individuals to eat. "
        "2. Rationale: Hunger is a biological signal that the body needs nutrients. This sensation prompts the behavior of eating to replenish energy and nutrients. "
        "3. Conclusion: It is reasonable to conclude that hunger causes eating. The answer is Yes. "
        "Final Answer: Yes"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'the number of ice cream sales' and 'the number of drowning incident'. "
        "Does 'the number of ice cream sales' causes 'the number of drowning incident'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Increases in ice cream sales and drowning incidents may occur simultaneously, especially during warmer months. "
        "2. Rationale: While both ice cream sales and drowning incidents might rise during the summer, this does not imply that one causes the other. Instead, both are likely correlated with a third factor, such as higher temperatures or increased outdoor activities during warm weather. "
        "3. Conclusion: The simultaneous increase in ice cream sales and drowning incidents is better explained by a common cause (warm weather) rather than a direct causal relationship between the two. The answer is No. "
        "Final Answer: No"},
    {"role": "user", "content":
        "This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
        "We are interested in the causal relationship between 'the number of ice cream sales' and 'the number of drowning incident'. "
        "Does 'the number of drowning incident' causes 'the number of ice cream sales'? "
        "Let's provide a step-by-step explanation, then give your final answer using yes or no."},
    {"role": "assistant", "content":
        "Step-by-Step Explanation: "
        "1. Observation: Drowning incidents do not have a direct impact on the demand or sales of ice cream. "
        "2. Rationale: The occurrence of drowning incidents is an unfortunate event that does not influence people's consumption behavior regarding ice cream. Similar to the previous explanation, any observed correlation is more likely due to external factors like seasonality rather than a direct causal link. "
        "3. Conclusion: There is no logical or direct pathway through which drowning incidents could cause an increase in ice cream sales. Any correlation observed is likely due to external, confounding variables. The answer is No. "
        "Final Answer: No"},
]

def create_prompt(cause, effect, model, tokenizer, shot_num):
    demos = demos_1
    prompt = {"role": "user", "content": f"This task is to determine the cause-and-effect relationship between two events based on commonsense knowledge. "
                                         f"We are interested in the causal relationship between '{cause}' and '{effect}'. "
                                         f"Does '{cause}' causes '{effect}'?"
                                         f"Let's provide a step-by-step explanation, then give your final answer using yes or no."
              }
    #demos.append(prompt)
    chat_template_models = ["mistral", "gemma", "llama"]
    if model.config.model_type in chat_template_models:
        input_str = tokenizer.apply_chat_template(demos[:shot_num*2]+[prompt], tokenize=False)
    elif model.config.model_type == "olmo":
        input_str = tokenizer.apply_chat_template(demos[:shot_num*2]+[prompt], tokenize=False)
    elif model.config.model_type == "falcon":
        #chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '>>QUESTION<<\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '>>ANSWER<<\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '>>ANSWER<<' }}\n{% endif %}\n{% endfor %}",
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '>>QUESTION<<\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '>>ANSWER<<\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '>>ANSWER<<' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(demos[:shot_num * 2] + [prompt], chat_template=chat_template, add_generation_prompt=True, tokenize=False)
    elif model.config.model_type == "bloom":
        chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'Question: \n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Answer: \n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Answer: ' }}\n{% endif %}\n{% endfor %}"
        input_str = tokenizer.apply_chat_template(demos[:shot_num * 2] + [prompt], chat_template=chat_template,
                                                  add_generation_prompt=True, tokenize=False)
    else:
        raise ValueError("Model type not supported")
    #input_str = input_str.replace("\t", "")
    return input_str


def openllm_pairwise_chat_yes_no(read_file, openllm_path=None, save_file=None, num_shot=3):
    # load open llm model
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(openllm_path)
    model = AutoModelForCausalLM.from_pretrained(openllm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    df = pd.read_csv(read_file)
    shot_num = num_shot
    # generate queries
    df[f'query_{shot_num}_demo'] = df.apply(lambda row: create_prompt(row['cause'], row['effect'], model, tokenizer, shot_num=shot_num), axis=1)
    # send queries to llm
    df['responses'] = df.progress_apply(
        lambda row: send_query_to_llm_chat(row[f"query_{shot_num}_demo"], model, tokenizer, num_return_sequences=10), axis=1)
    # extract answers
    df['extracted_answer'] = df.apply(lambda row: extract_answer_chat_yes_no(row['responses']), axis=1)
    # majority vote
    df['majority_answer'] = df.apply(lambda row: majority_vote(row['extracted_answer']), axis=1)

    df.to_csv(save_file, index=False)



parser = argparse.ArgumentParser(description='causality.')

# Add arguments
parser.add_argument('--read_file', type=str, help='file path to the dataset')
parser.add_argument('--save_file', type=str, help='save path to the dataset')
parser.add_argument('--openllm_path', type=str, help='path to model')
parser.add_argument('--num_shot', type=int, help='the number of demonstrated examples')

# Parse the arguments
args = parser.parse_args()

if args.read_file is not None and "meta-llama/Llama" not in args.openllm_path:
    openllm_pairwise_chat_yes_no(read_file=args.read_file,
                                 openllm_path= args.openllm_path,
                                 save_file=args.save_file,
                                 num_shot=args.num_shot)

