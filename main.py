import torch
import numpy as np
from models.VAE import VAE
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/machine_learning")
def machine_learning():
    return render_template("machine_learning.html")

@app.route("/machine_learning/vae")
def vae():
    model = VAE(2, 1)
    model.load_state_dict(torch.load("models/vae_model_weights.pth", map_location=torch.device('cpu')))
    model.eval()

    samples = model.sample(num_samples=1)
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

    return render_template("vae.html", image=pngImageB64String)

@app.route("/machine_learning/style_transfer")
def style_transfer():
    return render_template("style_transfer.html")


if __name__ == "__main__":
    app.run(debug=True)