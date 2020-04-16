from glob import glob
import json
from IPython.display import display
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pandas as pd

pd.set_option('display.width', None)

print("Number of train images: ", len(glob(f'../../train/*')))
print("Number of test images: ", len(glob(f'../../test/*')))

with open('../../iwildcam2020_train_annotations.json') as json_data:
    train_annotations = json.load(json_data)
    print(train_annotations.keys())

df_cat = pd.DataFrame(train_annotations["categories"])
# display(f"Total Categories: {df_cat.name.nunique()}")
# display(df_cat.sample(5))
#
# display("Samples of annotations and images")
df_train_annotations = pd.DataFrame(train_annotations["annotations"])
# display(df_train_annotations)
#
# display("images")
# df_train_images = pd.DataFrame(train_annotations["images"])
# display(df_train_images)
#
# print(train_annotations['info'])

with open('../../iwildcam2020_megadetector_results.json') as json_data:
    megadetector_results = json.load(json_data)
    print(megadetector_results.keys())
print(megadetector_results['info'])

df_detections = pd.DataFrame(megadetector_results["images"])
print(f'detection categories :\n {megadetector_results["detection_categories"]}')
print(f'detection output :\n {df_detections.head()}')

with open('../../iwildcam2020_test_information.json') as json_data:
    test_info = json.load(json_data)
    print(test_info.keys())

print(f'test images :\n {test_info["images"][0]}')


def draw_bbox(img_path):
    img_id = img_path.split('/')[-1].split('.')[0]
    img = mpimg.imread(img_path)
    detections = df_detections[df_detections.id == img_id].detections.values[0]
    annotation = df_train_annotations[df_train_annotations.image_id == img_id]

    count = annotation['count'].values
    cat_id = annotation.category_id
    cat = df_cat[df_cat.id == int(cat_id)].name.values[0]

    _ = plt.figure(figsize=(15, 20))
    _ = plt.axis('off')
    ax = plt.gca()
    ax.text(10, 100, f'{cat} {count}', fontsize=20, color='fuchsia')

    for detection in detections:
        # ref - https://github.com/microsoft/CameraTraps/blob/e530afd2e139580b096b5d63f0d7ab9c91cbc7a4/visualization/visualization_utils.py#L392
        x_rel, y_rel, w_rel, h_rel = detection['bbox']
        img_height, img_width, _ = img.shape
        x = x_rel * img_width
        y = y_rel * img_height
        w = w_rel * img_width
        h = h_rel * img_height

        cat = 'animal' if detection['category'] == "1" else 'human'
        bbox = patches.FancyBboxPatch((x, y), w, h, alpha=0.8, linewidth=6, capstyle='projecting', edgecolor='fuchsia',
                                      facecolor="none")

        ax.text(x + 1.5, y - 8, f'{cat} {detection["conf"]}', fontsize=10,
                bbox=dict(facecolor='fuchsia', alpha=0.8, edgecolor="none"))
        ax.add_patch(bbox)

    _ = plt.imshow(img)
    plt.show()

img_path = "../../train/8a00fe98-21bc-11ea-a13a-137349068a90.jpg"
draw_bbox(img_path)