# just does a basic check that 
# all the tasks are in a category
# and all the categories are a task

import json
import os

basedir = '/workspaces/data/tom/Totalsegmentator_MSD_format'
tasks = sorted(os.listdir(basedir))

# test categories
cats = json.loads(open('/workspaces/xmem_training/XMem/util/totalseg_categories.json', 'r').read())

for task in tasks:
    found = 0
    for cat in cats:
        if task in cats[cat]:
            found += 1

    assert found == 1, f"{task} found {found} times"


for cat in cats:
    for k in cats[cat]:
        assert k in tasks, f"{k} from {cat} not in tasks"

# test train test splits
cats = json.loads(open('/workspaces/xmem_training/XMem/util/totalseg_train_test_categories.json', 'r').read())

for task in tasks:
    found = 0
    for cat in cats:
        if task in cats[cat]['train']:
            found += 1
        if task in cats[cat]['test']:
            found += 1

    assert found == 1, f"{task} found {found} times"


for cat in cats:
    for split in cats[cat]:
        for k in cats[cat][split]:
            assert k in tasks, f"{k} from {cat} not in tasks"