import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from datetime import datetime

def make_train_test_split(img_names, split=0.2, seed=420420):
    n = len(img_names)
    k = int(n * split)
    random.seed(seed)
    test = set(random.sample(img_names, k))
    train = set([img_name for img_name in img_names if not img_name in test])

    return list(train), list(test)
    
def main(dirpath, make_splits=False, report_analytics=False, make_trimmed_splits=False, verbose=False):
    import shutil

    img_dir = os.path.join(dirpath, 'JPEGImages')
    msk_dir = os.path.join(dirpath, 'Annotations')
    train_img_dir = os.path.join(dirpath, 'train', 'JPEGImages')
    train_msk_dir = os.path.join(dirpath, 'train', 'Annotations')
    test_img_dir = os.path.join(dirpath, 'test', 'JPEGImages')
    test_msk_dir = os.path.join(dirpath, 'test', 'Annotations')
    trimmed_img_dir = os.path.join(dirpath, 'trimmed-test', 'JPEGImages')
    trimmed_msk_dir = os.path.join(dirpath, 'trimmed-test', 'Annotations')

    progress_file = os.path.join(dirpath, 'progress.txt')

    if make_splits:
        img_names = os.listdir(img_dir)
        train, test = make_train_test_split(img_names)

        iterator = tqdm(train, desc='copying over train files') if verbose else train
        for idx, img_name in enumerate(iterator):
            shutil.copytree(
                os.path.join(img_dir, img_name),
                os.path.join(train_img_dir, img_name)
            )
            shutil.copytree(
                os.path.join(msk_dir, img_name),
                os.path.join(train_msk_dir, img_name)
            )
            with open(progress_file, 'w') as f:
                f.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: Copied train {idx}/{len(iterator)}")

        iterator = tqdm(test, desc='copying over test files') if verbose else test
        for idx, img_name in enumerate(iterator):
            shutil.copytree(
                os.path.join(img_dir, img_name),
                os.path.join(test_img_dir, img_name)
            )
            shutil.copytree(
                os.path.join(msk_dir, img_name),
                os.path.join(test_msk_dir, img_name)
            )
            with open(progress_file, 'w') as f:
                f.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: Copied test {idx}/{len(iterator)}")

    if make_trimmed_splits:
        scans = os.listdir(train_img_dir)
        iterator = tqdm(scans, desc=f'trimming {dirpath}...') if verbose else scans
        for idx, scan in enumerate(iterator):

            frames = sorted(os.listdir(os.path.join(train_img_dir, scan)))
            masks = sorted(os.listdir(os.path.join(train_msk_dir, scan)))

            # find the non empty frames
            non_empty_frames = []
            for idx, (frame, mask) in enumerate(zip(frames, masks)):
                assert frame.split('.')[0] == mask.split('.')[0]
                mask_obj = cv2.imread(os.path.join(train_msk_dir, scan, mask))
                if np.any(mask_obj != 0):
                    non_empty_frames.append(idx)
            
            if len(non_empty_frames) == 0:
                breakpoint()
                
            # move the non empty frames into seperate folder
            os.makedirs(os.path.join(trimmed_img_dir, scan), exist_ok=True)
            os.makedirs(os.path.join(trimmed_msk_dir, scan), exist_ok=True)

            # pad idxs by 1 to include empty start and end frames
            start_idx = max(0, min(non_empty_frames) - 1)
            end_idx = min(len(frames)-1, max(non_empty_frames) + 1)

            for idx in range(start_idx, end_idx + 1):
                frame = frames[idx]
                mask = masks[idx]
                assert frame.split('.')[0] == mask.split('.')[0]

                shutil.copy2(
                    os.path.join(train_img_dir, scan, frame),
                    os.path.join(trimmed_img_dir, scan, frame)
                )
                shutil.copy2(
                    os.path.join(train_msk_dir, scan, mask),
                    os.path.join(trimmed_msk_dir, scan, mask)
                )

            with open(progress_file, 'w') as f:
                f.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: Trimmed {idx}/{len(iterator)}")


    task = [x for x in dirpath.split('/') if x.startswith('Task')][0]
    
    if report_analytics:
        print(f"N images in {task} / train: {len(os.listdir(train_img_dir))}")
        print(f"N images in {task} / test: {len(os.listdir(test_img_dir))}")

    return task, len(os.listdir(train_img_dir)), len(os.listdir(test_img_dir))


def plot_pies(results, colour_function, outpath='pie.png', ):

    # Extract labels and scores
    labels = list(results.keys())
    scores = list(results.values())

    # Plotting
    fig, ax = plt.subplots()
    fig.set_size_inches((10,10))

    # Absolute values
    wedges, text, autotext = ax.pie(scores, labels=labels, autopct='%1.1f%%')

    # Add dividing lines/borders
    for wedge in wedges:
        wedge.set_linewidth(1)  # Set the width of the dividing line
        wedge.set_edgecolor('white')  # Set the color of the dividing line

    # Color the wedges based on the output of the color_function
    colors = [colour_function(label) for label in labels]
    plt.setp(wedges, edgecolor='black', linewidth=1.5, facecolor='none')  # Set the default wedge style
    plt.setp([wedges[i] for i, color in enumerate(colors) if color], facecolor=(0.3,0.5,0.95))  # Set color for True labels

    # modify labels
    for i, label in enumerate(text):
        label.set_text(f"{labels[i]}\n{scores[i]} scans ({autotext[i].get_text()})")

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Medical decathlon test data distribution')
    txt = f"""{sum(scores)} total scans. 
    {sum([score for score, label in zip(scores, labels) if colour_function(label)])} are CTs, shown in blue."""
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(outpath)



if __name__ == '__main__':
    basedir = '/workspaces/data/tom/Totalsegmentator_MSD_format'
    tasks = sorted([os.path.join(basedir, p) for p in os.listdir(basedir)  if os.path.isdir(os.path.join(basedir, p))])
    APPROX_FILES_PER_TASK = 1000
    print(f'Making train test splits over {len(tasks)} tasks and thus {APPROX_FILES_PER_TASK * len(tasks)} datapoints')

    os.environ["RAY_DEDUP_LOGS"] = "0"
    import ray
    import numpy as np
    import time
    import warnings
    from ray.util.multiprocessing import Pool
    from tqdm.auto import tqdm
    ray.init()

    @ray.remote
    class Worker:
        def __init__(self, i): self.i = i

        def doSomeWork(self, task):
            _ = main(os.path.join(basedir, task, '480pVolumetricallyExtractedData'), make_splits=True, make_trimmed_splits=True, verbose=False)
            return task

    # start a pool and distribute jobs
    N_WORKERS = 70
    workers = [Worker.remote(i) for i in range(N_WORKERS)]
    handles = [workers[i % N_WORKERS].doSomeWork.remote(task) for i, task in enumerate(tasks)]
    results = [ray.get(handle) for handle in handles]