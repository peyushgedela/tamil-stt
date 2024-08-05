import os
import wave
import pyaudio
import time
import threading
from faster_whisper import WhisperModel
from collections import deque
import customtkinter as ctk
from tkinter.scrolledtext import ScrolledText

NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

def record_chunk(p, stream, file_path, chunk_length=3):  # Increased chunk length
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

def transcribe_chunk(model, audio_path):
    segments, _ = model.transcribe(audio_path, language="ta")
    return " ".join([segment.text for segment in segments])

def transcribe_and_update(model, p, stream, text_widget):
    recent_transcriptions = deque(maxlen=10)  # Keep only the 10 most recent transcriptions
    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file, chunk_length=3)  # Record for 3 seconds
            transcription = transcribe_chunk(model, chunk_file)
            os.remove(chunk_file)
            recent_transcriptions.append(transcription)
            
            # Update UI
            text_widget.config(state=ctk.NORMAL)
            text_widget.insert(ctk.END, f"{transcription}\n")
            text_widget.config(state=ctk.DISABLED)
            text_widget.see(ctk.END)
            
            time.sleep(1)  # Add a small delay to reduce CPU usage
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        with open("log.txt", "w", encoding="utf-8") as logfile:
            logfile.write(' '.join(recent_transcriptions))
        stream.stop_stream()
        stream.close()
        p.terminate()

def start_transcription():
    model_size = "base"
    model = WhisperModel(model_size, compute_type="int8")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    # Create a separate thread for transcription
    transcription_thread = threading.Thread(target=transcribe_and_update, args=(model, p, stream, text_widget))
    transcription_thread.daemon = True
    transcription_thread.start()

# Initialize customtkinter
ctk.set_appearance_mode("Dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (default), "green", "dark-blue"

# Create the main window
root = ctk.CTk()
root.title("Real-time Tamil Speech-to-Text Transcription")

# Create a ScrolledText widget
text_widget = ScrolledText(root, wrap=ctk.WORD, state=ctk.DISABLED, width=80, height=20)
text_widget.pack(padx=10, pady=10)

# Create a Start button
start_button = ctk.CTkButton(root, text="படியெடுத்தல்", command=start_transcription)
start_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
