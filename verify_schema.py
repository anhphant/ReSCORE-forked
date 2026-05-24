import json

with open('d:/GPA/cs221/doofans/ReSCORE-2gpu-vimqa/.temp/vimqa/dataset_examples/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('output_schema.txt', 'w', encoding='utf-8') as out:
    example = data[0]
    out.write(f"=== Example 0 ===\n")
    out.write(f"Keys: {list(example.keys())}\n")
    out.write(f"_id: {example['_id']}\n")
    out.write(f"question: {example['question']}\n")
    out.write(f"answer: {example['answer']}\n")
    out.write(f"type: {example['type']}\n")
    out.write(f"\ntype(context): {type(example['context'])}\n")
    out.write(f"len(context): {len(example['context'])}\n")
    out.write(f"\ntype(context[0]): {type(example['context'][0])}\n")
    out.write(f"len(context[0]): {len(example['context'][0])}\n")
    out.write(f"context[0][0] (title): {example['context'][0][0]}\n")
    out.write(f"type(context[0][1]): {type(example['context'][0][1])}\n")
    out.write(f"len(context[0][1]) (sentences): {len(example['context'][0][1])}\n")
    out.write(f"context[0][1][0] (first sentence): {example['context'][0][1][0]}\n")
    
    out.write(f"\ntype(supporting_facts): {type(example['supporting_facts'])}\n")
    out.write(f"len(supporting_facts): {len(example['supporting_facts'])}\n")
    out.write(f"supporting_facts[0]: {example['supporting_facts'][0]}\n")
    out.write(f"type(supporting_facts[0]): {type(example['supporting_facts'][0])}\n")
    out.write(f"supporting_facts[0][0] (title): {example['supporting_facts'][0][0]}\n")
    out.write(f"supporting_facts[0][1] (sent_id): {example['supporting_facts'][0][1]}\n")
    
    # Check a second example to confirm consistency
    example2 = data[1]
    out.write(f"\n=== Example 1 ===\n")
    out.write(f"Keys: {list(example2.keys())}\n")
    out.write(f"_id: {example2['_id']}\n")
    out.write(f"question: {example2['question']}\n")
    out.write(f"answer: {example2['answer']}\n")
    out.write(f"type: {example2['type']}\n")
    out.write(f"len(context): {len(example2['context'])}\n")
    out.write(f"len(supporting_facts): {len(example2['supporting_facts'])}\n")
    for i, sf in enumerate(example2['supporting_facts']):
        out.write(f"  sf[{i}]: title={sf[0]}, sent_id={sf[1]}\n")
    
    # Check test set to see if answer is missing
    with open('d:/GPA/cs221/doofans/ReSCORE-2gpu-vimqa/.temp/vimqa/dataset_examples/test.json', 'r', encoding='utf-8') as ft:
        test_data = json.load(ft)
    t0 = test_data[0]
    out.write(f"\n=== Test Example 0 ===\n")
    out.write(f"Keys: {list(t0.keys())}\n")
    out.write(f"Has 'answer': {'answer' in t0}\n")
    out.write(f"Has 'supporting_facts': {'supporting_facts' in t0}\n")
    
    # Count total examples per split
    with open('d:/GPA/cs221/doofans/ReSCORE-2gpu-vimqa/.temp/vimqa/dataset_examples/train.json', 'r', encoding='utf-8') as ft:
        train_data = json.load(ft)
    out.write(f"\n=== Dataset Counts ===\n")
    out.write(f"train: {len(train_data)}\n")
    out.write(f"dev: {len(data)}\n")
    out.write(f"test: {len(test_data)}\n")
