import pandas as pd
import os
import json

base_path = '/fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/CMSC828A-Spring2024/hw1/MMLU_data/test/'

list_of_files = os.listdir(base_path)

for fil_1 in list_of_files:

    test_1 = pd.read_csv(base_path + fil_1,names=['question', 'A', 'B', 'C', 'D', 'answer'])

    test_json = []

    for i,row in test_1.iterrows():

        base_instruction = 'The following are multiple choice questions (with answers) about ' + ' '.join(fil_1.split("_")[:-1]) + '.\n\n'

        question = row['question']

        choice='\nA.' + str(row['A']) + '\nB.' + str(row['B']) + '\nC.' + str(row['C']) + '\nD.' + str(row['D'])

        base_answer = '\nAnswer: '
        
        answer = row['answer']

        test_json.append({
            "instruction": base_instruction,
            "input": question + choice + base_answer,
            "output": answer
        })

    for fil_2 in list_of_files:

        if fil_1 != fil_2:

            test2_json = []

            test_2 = pd.read_csv(base_path + fil_2,names=['question', 'A', 'B', 'C', 'D', 'answer'])

            exemplar = ''

            for i,row in test_2.iterrows():

                base_instruction = 'The following are multiple choice questions (with answers) about ' + ' '.join(fil_2.split("_")[:-1]) + '.\n\n'

                question = row['question']

                choice='\nA.' + str(row['A']) + '\nB.' + str(row['B']) + '\nC.' + str(row['C']) + '\nD.' + str(row['D'])

                base_answer = '\nAnswer: '
                
                answer = row['answer']

                if i == 0:
                    exemplar += base_instruction
                 
                example = question + choice + base_answer + answer + '\n'

                exemplar += example

                if i == 4:
                    break

            for j, inst in enumerate(test_json):
                inst["instruction"] = exemplar + '\n' + test_json[j]["instruction"]
                test2_json.append(inst)


            with open('/fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/data/part_2/' + fil_1.strip('.csv') + '_' + fil_2.strip('.csv') + '.json', 'w') as f:
                json.dump(test2_json,f,indent = 6)

            print('File Saved!')


            

    

    