import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

from flask import (Flask, flash, render_template, redirect, request, session,
                   send_file, url_for)
from werkzeug.utils import secure_filename

# from utils import (is_allowed_file,
#                    make_thumbnail)

ALLOWED_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpeg', 'gif'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

cwd = os.getcwd()
print(cwd)

TORCH_MODEL_PATH = os.path.join(cwd, "model/resnet50_pytorch_st_dict.pth")
print(TORCH_MODEL_PATH)

print('Loading model......')

model_loaded = models.resnet50(pretrained=False).to(device)
model_loaded.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 3)).to(device)
model_loaded.load_state_dict(torch.load(TORCH_MODEL_PATH))
model_loaded.eval()

print('Model Successfully Loaded.....')


def is_allowed_file(filename):
    """ Checks if a filename's extension is acceptable """
    allowed_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and allowed_ext


def make_thumbnail(filepath):
    """ Converts input image to 128px by 128px thumbnail if not that size """
    img = Image.open(filepath)
    thumb = None
    w, h = img.size

    # if it is exactly 128x128, do nothing
    if w == 224 and h == 224:
        return True

    # if the width and height are equal, scale down
    if w == h:
        thumb = img.resize((224, 224), Image.BICUBIC)
        thumb.save(filepath)
        return True

    # when the image's width is smaller than the height
    if w < h:
        # scale so that the width is 128
        ratio = w / 224.
        w_new, h_new = 224, int(h / ratio)
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        top, bottom = 0, 0
        margin = h_new - 224
        top, bottom = margin // 2, 224 + margin // 2
        box = (0, top, 224, bottom)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True

    # when the image's height is smaller than the width
    if h < w:
        # scale so that the height is 128
        ratio = h / 224.
        w_new, h_new = int(w / ratio), 224
        thumb = img.resize((w_new, h_new), Image.BICUBIC)

        # crop the excess
        left, right = 0, 0
        margin = w_new - 224
        left, right = margin // 2, 224 + margin // 2
        box = (left, 0, right, 224)
        cropped = thumb.crop(box)
        cropped.save(filepath)
        return True
    return False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

app = Flask(__name__)
app.config['SECRET_KEY'] = "vivek"
app.config['UPLOAD_FOLDER'] = os.path.join(cwd, "UPLOAD")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']
        radio_box_val = request.form['options']
        print("radio_box_val")
        print(radio_box_val)

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # check if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)
            # HACK: Defer this to celery, might take time
            passed = make_thumbnail(filepath)
            if passed:
                return redirect(url_for('predict', filename=filename, radio_val = radio_box_val))
            else:
                return redirect(request.url)




@app.errorhandler(500)
def server_error(error):
    """ Server error page handler """
    return render_template('error.html'), 500


@app.route('/images/<filename>')
def images(filename):
    """ Route for serving uploaded images """
    if is_allowed_file(filename):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        flash("File extension not allowed.")
        return redirect(url_for('home'))


@app.route('/predict/<filename>/<radio_val>/')
def predict(filename, radio_val):
    """ After uploading the image, show the prediction of the uploaded image
    in barchart form
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_url = url_for('images', filename=filename)
    print('image_path')
    print(image_path)

    preds = F.softmax(model_loaded(data_transforms['validation'](Image.open(image_path)).to(device).resize_(1,3,224,224)),dim=1).cpu().data.numpy()
    print(preds)
    predictions = preds[0]
    print(predictions)

    vc_percent = str(round((predictions[1] * 100),2)) + str('%')
    w9_percent = str(round((predictions[2] * 100),2)) + str('%')

    if radio_val == "W9":
        print("W9 selected")
        class_percent_dict = "W9 Form: "+w9_percent
    elif radio_val == "VC":
        print("VC selected")
        class_percent_dict = "Void Check: " +vc_percent
    elif not radio_val:
        print("No button selected")
        class_percent_dict = "No class selected. Please select a class to validate."

    return render_template(
        'predict.html',
        class_percent=class_percent_dict,
        image_url=image_url
    )
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	# load_model()
	app.run(host='0.0.0.0')
