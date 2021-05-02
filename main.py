import torch
import numpy as np
from models.VAE import VAE
import io
import os
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/machine_learning")
def machine_learning():
    return render_template("machine_learning.html")

@app.route("/machine_learning/vae/get_image", methods=['POST', 'GET'])
def vae_get_image():
    z = np.random.normal(size=(1, 2))
    latent_dim01 = z[0, 0]
    latent_dim02 = z[0, 1]

    print("Called!!!")
    print(request.get_json())
    if request.method == 'POST':
        latent_dim01 = request.form['latent_dim01']
        latent_dim02 = request.form['latent_dim02']
        print(latent_dim01)
        print(latent_dim02)
        z = np.array([[latent_dim01, latent_dim02]])

    z = z.astype(np.float32)
    z = torch.from_numpy(z).to("cpu")
    model = VAE(2, 1)
    model.load_state_dict(torch.load("models/vae_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    samples = model.decode(z)
    samples = samples.detach().numpy()

    # Generate plot
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.imshow(np.ones((28, 28, 1)) - np.reshape(samples[0], (28, 28, 1)), cmap='Greys')

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    print("Before return!!")
    #return pngImageB64String, latent_dim01, latent_dim02
    return jsonify({'image': pngImageB64String, 'latent_dim01': latent_dim01, 'latent_dim02': latent_dim02, 'success': True})

@app.route("/machine_learning/vae", methods=['POST', 'GET'])
def vae():
    print("Faaaaaalsch!!")
    vae_result = vae_get_image().get_json()
    return render_template("vae.html", image=vae_result['image'], latent_dim01=vae_result['latent_dim01'], latent_dim02=vae_result['latent_dim02'])

@app.route("/machine_learning/style_transfer")
def style_transfer():
    return render_template("style_transfer.html")


if __name__ == "__main__":
    app.run(debug=True)