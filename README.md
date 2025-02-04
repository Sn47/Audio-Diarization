

# 📄 Audio-Diarization

## 1️⃣ Introduction
This project focuses on extracting and processing audio from video files using Python in a Jupyter Notebook environment. It provides step-by-step instructions for audio stripping, transformation, and visualization. Audio extraction is essential for applications like speech analysis, music processing, and multimedia content manipulation.


---

## 2️⃣ Objectives
- Extract audio from video files.
- Save extracted audio in various formats.
- Analyze and visualize audio signals.
- Explore different audio processing techniques.
- Provide efficient, reusable methods for future projects in audio and video analysis.


## 3️⃣ Methodology

### 3.1 Environment Setup
The project is developed using Python in a Jupyter Notebook. The following libraries are required for implementation:

```bash
pip install ffmpeg-python librosa numpy matplotlib pydub pyannote.audio
```

- **ffmpeg-python**: For audio extraction from video files.
- **librosa**: For audio processing and visualization.
- **matplotlib**: For visualizing the audio waveform and spectrogram.
- **pydub**: For additional audio processing functionalities (e.g., format conversion).
- **pyannote.audio**: A specialized library for speaker diarization.

---

### 3.2 Audio Extraction
To extract audio from a video file, the **FFmpeg** library is used:

```python
import ffmpeg

input_file = "input_video.mp4"
output_audio = "output_audio.wav"
ffmpeg.input(input_file).output(output_audio).run()
```

This method allows you to extract audio in the desired format (e.g., WAV or MP3) without losing quality.

---

### 3.3 Error Handling
To ensure smooth execution, exceptions are added to handle audio extraction failures:

```python
try:
    ffmpeg.input(input_file).output(output_audio).run()
    print("Audio extraction successful!")
except Exception as e:
    print(f"Error extracting audio: {e}")
```

---

### 3.4 Comparison with Other Methods
Other methods for audio extraction include:
- **MoviePy**: A Python-based alternative offering a higher-level API:

  ```python
  from moviepy.editor import VideoFileClip
  clip = VideoFileClip(input_file)
  clip.audio.write_audiofile(output_audio)
  ```

- **Pydub**: Allows for additional functionalities like trimming and format conversion.

  ```python
  from pydub import AudioSegment
  audio = AudioSegment.from_file(input_file)
  audio.export(output_audio, format="wav")
  ```

---

### 3.5 Audio Processing

#### 3.5.1 Loading and Visualizing Audio
Once the audio is extracted, it is visualized using **Librosa**:

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_data, sr = librosa.load(output_audio, sr=None)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio_data, sr=sr)
plt.title("Audio Waveform")
plt.show()
```

#### 3.5.2 Spectrogram Analysis
To analyze the frequency content over time, a **spectrogram** is generated:

```python
import numpy as np
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.show()
```

![Screenshot 2025-02-04 at 10 05 01 PM](https://github.com/user-attachments/assets/672d2850-f333-4765-b79e-35b1678887fc)  
*Figure 3: Spectrogram displaying frequency components over time.*

---

### 3.6 Speaker Diarization: Identifying Person 1 and Person 2
For analyzing multiple speakers in the audio, **Speaker Diarization** is applied using the **pyannote.audio** library. This process allows us to segment the audio based on speaker changes and identify which segments belong to Person 1 or Person 2.
![Screenshot 2025-02-04 at 10 20 20 PM](https://github.com/user-attachments/assets/8c3a3780-281e-4171-b215-b2969ce2f3bb)

#### 3.6.1 Speaker Diarization
We use the **pyannote.audio** library to perform speaker diarization and label segments by speaker:

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Apply diarization on the extracted audio file
diarization = pipeline({'uri': 'filename', 'audio': output_audio})

# Display the diarization result
for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {speech_turn.start:.2f}s to {speech_turn.end:.2f}s")
```

Explanation:
- **`pyannote.audio`** is used to identify distinct speakers in the audio, labeling them as `SPEAKER_00`, `SPEAKER_01`, etc.
- The output is a series of time intervals (start and end) for each speaker, allowing us to know when each person speaks.
![Screenshot 2025-02-04 at 10 21 06 PM](https://github.com/user-attachments/assets/f42a4c31-092a-4fa0-83af-175be3da3282)

#### 3.6.2 Visualization of Speakers
We visualize the segments for Person 1 and Person 2 as follows:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 3))
colors = ['blue', 'green']  # Different colors for different speakers
labels = {'blue': 'Person 1', 'green': 'Person 2'}

# Draw speaker segments with different colors for different speakers
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = turn.start
    end = turn.end
    color = colors[int(speaker[-2:]) % len(colors)]
    rect = patches.Rectangle((start, 0), end - start, 1, linewidth=1, edgecolor='none', facecolor=color)
    ax.add_patch(rect)

# Set up the plot
ax.set_xlabel('Time (s)')
ax.set_title('Speaker Diarization: Person 1 and Person 2')
ax.set_yticks([])  # No y ticks, as it's only time-based
ax.legend(handles=[patches.Patch(color=color, label=label) for color, label in labels.items()], loc='upper right')
ax.grid(True)
plt.show()
```

Explanation:
- **Person 1** and **Person 2** are visually represented with different colors (blue and green, respectively).
- The segments where each person is speaking are plotted as colored blocks on the time axis.
![Screenshot 2025-02-04 at 10 21 32 PM](https://github.com/user-attachments/assets/f09e4cef-7ac9-417a-9c99-dd28505777a8)

#### 3.6.3 Speech and Non-Speech Segments
In addition to identifying the speakers, the **Voice Activity Detection (VAD)** algorithm is used to detect periods of speech and non-speech. These periods are visualized as red (non-speech) and green (speech).

```python
from pyannote.audio.pipelines import VoiceActivityDetection

# Use VAD to detect speech and non-speech segments
vad_pipeline = VoiceActivityDetection.from_pretrained("pyannote/voice-activity-detection")

# Detect and display speech segments
speech_segments = list(vad_pipeline({'uri': 'filename', 'audio': output_audio}).itertracks())

# Plotting speech and non-speech segments
for segment in speech_segments:
    color = 'green' if segment.is_speech else 'red'
    ax.add_patch(patches.Rectangle((segment.start, 0), segment.end - segment.start, 1, linewidth=1, edgecolor='none', facecolor=color))
```

This allows the distinction between **speech** (green) and **non-speech** (red) to be visualized alongside the speaker diarization.
![Screenshot 2025-02-04 at 10 22 54 PM](https://github.com/user-attachments/assets/4cd5f9fd-9cef-42a9-b805-26bd5e5601a7)


Final with speech , non-speech , speaker 1 , speaker 2 and overlapping too
![Screenshot 2025-02-04 at 10 23 24 PM](https://github.com/user-attachments/assets/b0940d12-2b89-449c-8a72-cd98f9a6d4fe)


___

### 3.7 Advanced Audio Processing Techniques
Advanced processing methods applied to audio include:
- **Filtering**: Applying filters to isolate or remove frequencies.
- **Pitch Detection**: Analyzing the pitch of audio signals.
- **Time-Stretching**: Modifying the audio duration without altering pitch.

___

## 4️⃣ Results and Analysis

The extracted audio file is successfully saved and visualized. The waveform and spectrogram provide insights into amplitude and frequency variations over time.

### 4.1 Observations
- **Waveform**: The waveform reveals amplitude changes over time, indicating speech, music, or silence.
- **Spectrogram**: The spectrogram highlights the frequency components of the audio, aiding in the identification of dominant frequencies.

---

### 4.2 Performance Insights
- **Processing Speed**: FFmpeg is highly efficient and faster than MoviePy for large files.
- **Memory Usage**: Visualizing large audio files can be memory-intensive; careful handling of large files is advised.
- **Accuracy**: The quality of extracted audio depends on the video's compression and encoding.

---

## 5️⃣ Performance Considerations

- **Processing Speed**: FFmpeg is faster than MoviePy for larger files, while MoviePy provides an easier interface for basic usage.
- **Memory Usage**: Visualizations for large audio files require careful memory management.
- **Accuracy**: FFmpeg extracts high-quality audio; other methods may introduce slight losses.

---

## 6️⃣ Use Cases & Applications

- **Speech Analysis**: Use audio for transcription or speaker diarization.
- **Music Processing**: Analyze beats, tempo, or genre classification.
- **Noise Detection**: Apply noise reduction to enhance speech clarity.
- **Forensics & Surveillance**: Analyze audio evidence for legal cases.
- **Media Editing**: Manipulate audio for podcasts, interviews, or content creation.


---

## 7️⃣ Conclusion

This project demonstrates how to extract and analyze audio from video files using Python. The techniques presented form a foundation for more advanced audio processing applications, including speech recognition, noise reduction, and machine learning-based classification.

---

## 8️⃣ Future Work

- **Noise Reduction**: Implement spectral gating or deep learning models for noise reduction.
- **Machine Learning**: Apply models for audio classification (e.g., speech-to-text, emotion recognition).
- **Batch Processing**: Automate processing for multiple video files, especially for large datasets.
- **Real-time Audio Processing**: Implement real-time extraction and transcription for live applications.

---

📌 **Author:** [Ahsan Naveed]  
📆 **Date:** [2024-07-15]

