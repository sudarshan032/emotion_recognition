import wavio 
import numpy as np
import sounddevice as sd

FILE_TYPES = set(["wav", "WAV"])
recording = False
record_thread = None
record_data = []
output_file = "audio_files/output.wav"

# Audio recording functions
def record_audio(filename, fs):
    global recording, record_data
    recording = True
    record_data = []
    try:
        print("Recording...")
        with sd.InputStream(samplerate=fs, channels=2, dtype='int16') as stream:
            while recording:
                data, _ = stream.read(1024)
                record_data.append(data)
    except Exception as e:
        print(f"Error recording audio: {e}")

    if record_data:
        audio_data = np.concatenate(record_data, axis=0)
        wavio.write(filename, audio_data, fs, sampwidth=2)
        print(f"Audio saved as {filename}")






@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, record_thread
    if not recording:
        recording = True
        record_thread = threading.Thread(target=record_audio, args=(output_file, 16000))
        record_thread.start()
        return jsonify({"status": "recording"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    if recording:
        recording = False
        record_thread.join()
        return jsonify({"status": "finished"})
    return jsonify({"status": "not recording"})
