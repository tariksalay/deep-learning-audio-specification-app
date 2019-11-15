from flask import Flask
from WebApp.main_page import main_page
from MachineLearning.machinelearning import machinelearning

UPLOAD_FOLDER = 'C:/Users/tondi/OneDrive/Documents/GitHub/CSEE5590-IOT-Robotics/ICP 9/downloads'
ALLOWED_EXTENSIONS = set(['txt', 'wav'])
app = Flask(__name__)
app.secret_key = 'secret key'
app.register_blueprint(main_page)
app.register_blueprint(machinelearning, url_prefix='/machinelearning')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if __name__ == "__main__":
    app.run(Debug=False)
