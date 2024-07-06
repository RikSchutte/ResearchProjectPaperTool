import numpy as np

import joblib
import torch

import config
import dataset
from model import EntityModel

def get_keys_by_value(d, target_values):
    # Ensure target_values is a set for efficient membership testing
    if isinstance(target_values, int):
        target_values = {target_values}
    else:
        target_values = set(target_values)
    
    # Find keys where any of the target_values are in the corresponding list
    matching_keys = [key for key, values in d.items() if any(value in target_values for value in values)]
    
    return matching_keys


if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = """
    In the end, 139 participants participated, of which 49 (40%) were male and 30 (63%) were female.
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    print(sentence)
    sentence = sentence.split()
    # print(tokenized_sentence)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("mps")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _, _ = model(**data)


    label_list = enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]
    
    # print(label_list, len(label_list))
    mask = np.array(label_list != 'o', dtype=bool)
    masked_tokens = list(np.array(tokenized_sentence)[mask][1:-1])


    token_dict = {x : config.TOKENIZER.encode(x, add_special_tokens=False) for x in sentence}
    result = get_keys_by_value(token_dict, masked_tokens)
    if len(result) != 1:
        print('The N-value could not be determined \n')
    else:
        print(f'The found N-Value is: {result[0]} \n')