from flask import Flask, render_template, request, jsonify
from data_handler import save_fingerprint

DEBUG = False

app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return render_template("cpu_gpu_fp2.html")


@app.route('/get_fingerprint', methods=['POST'])
def get_offscreen_fingerprint():
    cpu_fingerprint = request.form["cpu_fingerprint"]
    gpu_fingerprint = request.form["gpu_fingerprint"]
    device_name = request.form["device_name"]
    cur_time = request.form["cur_time"]
    save_fingerprint(device_name, cpu_fingerprint, gpu_fingerprint, cur_time)
    return jsonify({"code":200, "msg":"fingerprint received"})


@app.route('/get_starttime', methods=['POST'])
def get_starttime():
    start_time = request.form["start_time"]
    print(f"[{start_time}] Start Extracting Data\n")
    return jsonify({"code":200, "msg":"start time received"})


if __name__ == "__main__":
    logo = '''ooooo                                  .o8   .oooooo..o oooo
`888'                                 "888  d8P'    `Y8 `888
 888          .ooooo.   .oooo.    .oooo888  Y88bo.       888 .oo.    .ooooo.  oooo oooo    ooo
 888         d88' `88b `P  )88b  d88' `888   `"Y8888o.   888P"Y88b  d88' `88b  `88. `88.  .8'
 888         888   888  .oP"888  888   888       `"Y88b  888   888  888   888   `88..]88..8'
 888       o 888   888 d8(  888  888   888  oo     .d8P  888   888  888   888    `888'`888'
o888ooooood8 `Y8bod8P' `Y888""8o `Y8bod88P" 8""88888P'  o888o o888o `Y8bod8P'     `8'  `8'
'''
    print (logo)
    if DEBUG:
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        from gevent import pywsgi
        server = pywsgi.WSGIServer(('0.0.0.0', 8088), app, log=None)
        server.serve_forever()
