import pickle
from vizualize_results import viz, viz_multiview
from wandb_utils.wandb import WandB
import pdb

with open('/home/stephen/Documents/3DDet/output/kitti_models/pointpillar/default/eval/epoch_20/val/default/result.pkl', 'rb') as j:
    x = pickle.load(j)
labels = []
centroids = []
dims = []
# id = '000008'
count_limit = 3
count = 0
# pdb.set_trace()
for scene in x:
    if count < count_limit:
        centroids = scene['location']
        dims = scene['dimensions']
        labels = scene['name']
        id = scene['frame_id']
        # pdb.set_trace()
        for row in centroids:
            row = list(row)
        for row1 in dims:
            row1 = list(row1)
        for row2 in labels:
            row2 = list(row2)
        # print("centroids: ", centroids)
        labels = list(labels)

        ### VIZUALIZE Pred
        viz(centroids, dims, labels, id)

        # ### VIZUALIZE Ground Truth Version
        # run = WandB(
        #     project='3DDet_Viz',
        #     enabled=True,
        #     log_objects=True,
        #     name='GT Test Viz: ' + id
        # )
        # viz_multiview(id, run)
        # count = count + 1
        # pdb.set_trace()
        # run.finish()
