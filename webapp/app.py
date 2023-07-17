from flask import Flask, request, render_template, jsonify
import parselmouth as pm


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def process_audio():
    print('Server endpoint "/upload" has been triggered')
    
    # Process the uploaded data or perform any required actions
    audio_data_str = request.get_data('audio')
    list_audio_data = audio_data_str.split(',')
    audio_data = [float(i) for i in list_audio_data]

    # Perform analysis using Praat-Parselmouth
    snd = pm.Sound(audio_data)
    # save to wav file
    snd.save('audio.wav', 'WAV')

    # Create a response dictionary with the analyzed waveform data
    response_data = {
        'waveform': 'ok',
    }

    # Return the response as JSON
    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
