import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import numpy as np
import pandas as pd
import os
import keras.backend as K
from keras.applications import VGG16
from keras.preprocessing.image import load_img, save_img, img_to_array
from PIL import Image
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


# Initialize the Flask application
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'

app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    
    img_height = 1024
    img_width = 1024
    img_size = img_height * img_width
    img_channels = 3
    CONTENT_IMAGE_POS = 0
    STYLE_IMAGE_POS = 1
    GENERATED_IMAGE_POS = 2

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

    def process_img(path):
        img = Image.open(path)
        img = img.resize((img_width, img_height))

        # Convert image to data array
        data = np.asarray(img, dtype='float32')
        data = np.expand_dims(data, axis=0)
        data = data[:, :, :, :3]

        # Apply pre-process to match VGG16 we are using
        data[:, :, :, 0] -= 103.939
        data[:, :, :, 1] -= 116.779
        data[:, :, :, 2] -= 123.68

        # Flip from RGB to BGR
        data = data[:, :, :, ::-1]

        return data


    def get_layers(content_matrix, style_matrix, generated_matrix):
        input_tensor = K.concatenate([content_matrix, style_matrix, generated_matrix], axis=0)
        model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # Convert layers to dictionary
        layers = dict([(layer.name, layer.output) for layer in model.layers])

        # Pull the specific layers we want
        c_layers = layers['block2_conv2']
        s_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        s_layers = [layers[layer] for layer in s_layers]

        return c_layers, s_layers

    def content_loss(content_features, generated_features):
        """
        Computes the content loss
        :param content_features: The features of the content image
        :param generated_features: The features of the generated image
        :return: The content loss
        """
        return 0.5 * K.sum(K.square(generated_features - content_features))

    def gram_matrix(features):

        return K.dot(features, K.transpose(features))

    def style_loss(style_matrix, generated_matrix):
        style_features = K.batch_flatten(K.permute_dimensions(style_matrix, (2, 0, 1)))
        generated_features = K.batch_flatten(K.permute_dimensions(generated_matrix, (2, 0, 1)))

        # Get the gram matrices
        style_mat = gram_matrix(style_features)
        generated_mat = gram_matrix(generated_features)

        return K.sum(K.square(style_mat - generated_mat)) / (4.0 * (img_channels ** 2) * (img_size ** 2))

    def variation_loss(generated_matrix):
        a = K.square(generated_matrix[:, :img_height-1, :img_width-1, :] - generated_matrix[:, 1:, :img_width-1, :])
        b = K.square(generated_matrix[:, :img_height-1, :img_width-1, :] - generated_matrix[:, :img_height-1, 1:, :])

        return K.sum(K.pow(a + b, 1.25))


    def total_loss(c_layer, s_layers, generated):
        content_weight  =0.025
        style_weight = 1.0
        variation_weight = 1.0

        # Content loss
        content_features = c_layer[CONTENT_IMAGE_POS, :, :, :]
        generated_features = c_layer[GENERATED_IMAGE_POS, :, :, :]
        c_loss = content_loss(content_features, generated_features)

        # Style loss
        s_loss = None
        for layer in s_layers:
            style_features = layer[STYLE_IMAGE_POS, :, :, :]
            generated_features = layer[GENERATED_IMAGE_POS, :, :, :]
            if s_loss is None:
                s_loss = style_loss(style_features, generated_features) * (style_weight / len(s_layers))
            else:
                s_loss += style_loss(style_features, generated_features) * (style_weight / len(s_layers))

        # Variation loss (for regularization)
        v_loss = variation_loss(generated)

        return content_weight * c_loss + s_loss + variation_weight * v_loss



    def eval_loss_and_grads(generated):
        generated = generated.reshape((1, img_height, img_width, 3))
        outs = f_outputs([generated])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values



    def save_image(filename, generated):
        generated = generated.reshape((img_height, img_width, 3))
        generated = generated[:, :, ::-1]
        generated[:, :, 0] += 103.939
        generated[:, :, 1] += 116.779
        generated[:, :, 2] += 123.68

        generated = np.clip(generated, 0, 255).astype('uint8')

        imsave(filename, Image.fromarray(generated))


    class Evaluator(object):

        def __init__(self):
            self.loss_value = None
            self.grad_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values



    
    # Get the name of the uploaded files
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)

    content_path = filenames[0]
    style_path = filenames[1]

    print(content_path)
    print(style_path)

    generated_img = np.random.uniform(0, 255, (1, img_height, img_width, 3)) - 128.
    content = process_img(content_path)
    style = process_img(style_path)

    content_image = K.variable(content)
    style_image = K.variable(style)
    generated_image = K.placeholder((1, img_height, img_width, 3))
    loss = K.variable(0.)

    content_layer, style_layers = get_layers(content_image, style_image, generated_image)

    loss = total_loss(content_layer, style_layers, generated_image)
    grads = K.gradients(loss, generated_image)

    evaluator = Evaluator()
    iterations = 10

    outputs = [loss]
    outputs += grads
    f_outputs = K.function([generated_image], outputs)

    for i in range(2):
        print('Iteration:', i)
        start_time = time.time()
        generated_img, min_val, info = fmin_l_bfgs_b(evaluator.loss, generated_img.flatten(),
                                                     fprime=evaluator.grads, maxfun=20)
        print('Loss:', min_val)
        end_time = time.time()
        print('Iteration {} took {} seconds'.format(i, end_time - start_time))
        name = 'uploads/{}-{}{}'.format(1, i, ".jpg")
        save_image(name,generated_img)
        print('Saved image to: {}'.format(name))
    return render_template('upload.html', filenames=filenames)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("4555"),
        debug=True
    )