import io
import parselmouth as pm
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import json
import pandas as pd

# AudioSegment.converter = "C:\\PATH_programs\\ffmpeg.exe"
# AudioSegment.ffmpeg = "C:\\PATH_programs\\ffmpeg.exe"
# AudioSegment.ffprobe = "C:\\PATH_programs\\ffprobe.exe"


def convert_webm_to_wav(input_file, output_file):
    try:
        # Load the WebM audio file using pydub
        audio = AudioSegment.from_file(input_file, format="webm")
        # Export the audio to WAV format
        audio.export(output_file, format="wav")

        print(f"Conversion successful. Output file saved as {output_file}")

    except Exception as e:
        print("An error occurred during conversion:", str(e))


def run_performance_report(audio_bytes: bytes) -> dict:
    print("run_performance_report() called")

    # save to webm file
    with open("webapp/testing/webm/audio.webm", "wb") as f:
        f.write(audio_bytes)

    # convert webm to wav
    convert_webm_to_wav(
        "webapp/testing/webm/audio.webm", "webapp/testing/wav/audio.wav"
    )

    # load wav file
    snd = MySound("webapp/testing/wav/audio.wav")
    performance_report = snd.getPerformanceReport()
    return performance_report


class MySound(pm.Sound):
    """MySound is a subclass of pm.Sound that adds a few methods to the pm.Sound class."""

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self = self.convert_to_mono()  # convert to mono if necessary

    def getWaveform(self):
        """Returns the waveform of the sound as a numpy array."""
        return {"x": self.xs().tolist(), "y": self.values[0].T.tolist()}

    def getSpectrogram(self):
        """Returns the spectrogram of the sound as a numpy array."""
        spectrogram = self.to_spectrogram()
        x, y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        return {"x": x.tolist(), "y": y.tolist(), "values": sg_db.tolist()}

    def getIntensity(self):
        """Returns the intensity of the sound as a numpy array."""
        intensity = self.to_intensity()
        return {"x": intensity.xs().tolist(), "y": intensity.values[0].T.tolist()}

    def getIntensityInfo(self):
        """Returns the information about the intensity of the sound as a tuple (mean, std, max)."""
        intensity = self.to_intensity()
        intensity_values = intensity.values[0].T
        intensity_mean = np.mean(intensity_values)
        intensity_std = np.std(intensity_values)
        intensity_max = np.max(intensity_values)

        return intensity_mean, intensity_std, intensity_max

    def getPitch(self):
        """Returns the pitch of the sound as a numpy array."""
        pitch = self.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values[pitch_values == 0] = np.nan
        return {"x": pitch.xs().tolist(), "y": pitch_values.tolist()}

    def getPitchInfo(self):
        """Returns the information about the pitch of the sound as a tuple (mean, std, min, max)."""
        pitch = self.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values[pitch_values == 0] = np.nan
        # remove nan values
        pitch_values = pitch_values[~np.isnan(pitch_values)]
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_min = np.min(pitch_values)
        pitch_max = np.max(pitch_values)

        return pitch_mean, pitch_std, pitch_min, pitch_max

    def getSpeechOnly(self):
        """Returns a subset of the sound that contains only the speech portion of the sound."""
        waveform = pd.Series(data=self.values[0].T, index=self.xs())
        intensity = pd.Series(
            data=self.to_intensity().values[0].T, index=self.to_intensity().xs()
        )
        mean_int = intensity.mean()

        # find segments with intensity below mean
        segments = []
        start = None
        for t, a in zip(intensity.index, intensity.values):
            if a < mean_int:
                if start is None:
                    start = t
            else:
                if start is not None:
                    segments.append((start, t))
                    start = None

        for start, end in segments:
            waveform = waveform[~((waveform.index >= start) & (waveform.index <= end))]
        extracted_sound = MySound(
            waveform.values.T, sampling_frequency=self.sampling_frequency
        )

        return extracted_sound

    def getPerformanceReport(self):
        """Returns a dictionary containing the performance report of the sound."""
        # get duration
        duration = self.duration
        # get speech only
        speech_only = self.getSpeechOnly()
        speech_duration = speech_only.duration
        ratio_speech_time = speech_duration / duration

        # we know perform the analysis on the speech only part of the sound
        intensity_mean, intensity_std, intensity_max = speech_only.getIntensityInfo()
        pitch_mean, pitch_std, pitch_min, pitch_max = speech_only.getPitchInfo()

        report = {
            "duration": duration,
            "speech_duration": speech_duration,
            "ratio_speech_time": ratio_speech_time,
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "intensity_max": intensity_max,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
        }

        return report
