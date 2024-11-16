from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # The image is now served from the /static folder
    image_url = 'static/output/original_with_boxes_1-normal-lumbar-spine-mri-living-art-enterprises.jpg'
    return render_template('index.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
