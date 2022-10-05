from pathlib import Path
import requests
import json
from PIL import Image

def cache_dataset(json_file, dataset_dir):
    '''Checks if all images in json are present in dataset_dir. If not, downloads them. Also copies json into dataset_dir. Overwrites existing json.

    Keyword arguments:
    json_file -- path to json
    dataset_dir -- path where dataset is saved to
    '''

    img_dir = Path().joinpath(dataset_dir, 'images')
    if not img_dir.exists():
        img_dir.mkdir()

    json_file = Path(json_file)
    dest = Path().joinpath(dataset_dir, json_file.name)
    src = Path(json_file)
    dest.write_text(src.read_text()) #for binary files

    with open(json_file) as json_file:
        json_file = json.load(json_file)
        for x in json_file:
            p = Path(dataset_dir) / 'images' / x['imgName']
            corrupt = True
            while corrupt:
                corrupt = False
                if p.is_file():
                    try:
                        Image.open(p).convert("RGB")
                    except (IOError, SyntaxError):
                        p.unlink()
                        corrupt = True
                        print(f'Image {p} was corrupt. Redownload')
                if not p.is_file():
                    via_data = json.loads(x['viaProject'])
                    r = requests.get(via_data['file']['1']['originalSrc'], verify=False)
                    with open(Path().joinpath(img_dir, x['imgName']), "wb") as f:
                        f.write(r.content)

    print('done!')
