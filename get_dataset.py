import os
import glob
import json
from PIL import Image

idx = 0
for page in range(70):
    files = glob.glob('./ffhq-dataset/images1024x1024/'+str(page).zfill(2)+'000/*.png')

    for f in files:
        filename = os.path.basename(f)
        json_file = open('./ffhq-features-dataset/json/'+filename[0:5]+'.json')
        json_obj = json.load(json_file)
        
        try:
            face = json_obj[0]['faceAttributes']
        except:
            print('json miss')
            continue

        if face['gender'] != 'female': 
            continue
        if face['age'] < 10 or face['age'] > 40:
            continue
        if face['glasses'] != 'NoGlasses':
            continue

        img = Image.open(f)
        img_resize = img.resize((256, 256))
        img_resize.save('./ffhq-dl21/'+filename)
        
        idx += 1
        if idx % 100 == 0:
            print(idx)
