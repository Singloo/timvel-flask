from flask import Flask, Response, request, json
import requests
import os
from os import path
from io import BytesIO
from fastai.vision import load_learner, open_image
import asyncio
app = Flask(__name__)
dir = path.dirname(__file__)

face_recognizer = load_learner(
    path.join(dir, './models'), file='face_recognizer.pkl')

classes = ['5_o_Clock_Shadow',
           'Arched_Eyebrows',
           'Attractive',
           'Bags_Under_Eyes',
           'Bald',
           'Bangs',
           'Big_Lips',
           'Big_Nose',
           'Black_Hair',
           'Blond_Hair',
           'Blurry',
           'Brown_Hair',
           'Bushy_Eyebrows',
           'Chubby',
           'Double_Chin',
           'Eyeglasses',
           'Goatee',
           'Gray_Hair',
           'Heavy_Makeup',
           'High_Cheekbones',
           'Male',
           'Mouth_Slightly_Open',
           'Mustache',
           'Narrow_Eyes',
           'No_Beard',
           'Oval_Face',
           'Pale_Skin',
           'Pointy_Nose',
           'Receding_Hairline',
           'Rosy_Cheeks',
           'Sideburns',
           'Smiling',
           'Straight_Hair',
           'Wavy_Hair',
           'Wearing_Earrings',
           'Wearing_Hat',
           'Wearing_Lipstick',
           'Wearing_Necklace',
           'Wearing_Necktie',
           'Young']


def _respJson(obj):
    return Response(json.dumps(obj), content_type="application/json; charset=utf-8")


def get_bytes(url):
    response = requests.get(url)
    return response.content


@app.route('/', methods=['GET'])
def hello():
    return app.send_static_file('/static/index.html')


@app.route('/face/recognize', methods=['POST'])
def recognize():
    bytes = get_bytes(request.get_json()["url"])
    print('aaa')
    img = open_image(BytesIO(bytes))
    print('bbb')
    pred_classes, _, losses = face_recognizer.predict(img)
    print(pred_classes, type(pred_classes))
    return _respJson({
        # "predictions": pred_classes
        "predictions": sorted(
            zip(classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)
    print('app is running at 0.0.0.0:5000')
