import json
import os
import random

cats = json.loads(open('/workspaces/xmem_training/XMem/util/totalseg_categories.json', 'r').read())

train_test_split = {}

for cat, tasks in cats.items():
    n_train = 2 * len(tasks) // 3
    train_tasks = random.sample(tasks, n_train)
    test_tasks = [task for task in tasks if not task in train_tasks]

    train_test_split[cat] = {
        'train': train_tasks,
        'test': test_tasks
    }

with open('/workspaces/xmem_training/XMem/util/totalseg_train_test_categories.json', 'w') as f:
    f.write(json.dumps(train_test_split))
