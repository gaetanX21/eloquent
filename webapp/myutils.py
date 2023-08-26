import io
import parselmouth as pm
import numpy as np
import soundfile as sf
import json
import pandas as pd
import speech_recognition as sr
import ffmpeg
import subprocess
import nltk
from nltk.tokenize import word_tokenize
from constants import *
import matplotlib.pyplot as plt


def convert_to_wav(
    input_audio_bytes: bytes,
    numberOfChannels: int,
    sample_rate: int,
    sample_size: int,
    mime: str,
) -> bytes:
    # Decode the audio stream from the original file (which is probably in webm format)
    format_ = mime.split("/")[1]
    acodec = f"pcm_s{sample_size}le"
    ac = numberOfChannels
    ar = sample_rate

    # Construct FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        "pipe:0",  # Input from pipe
        "-f",
        "wav",  # Output format
        "-acodec",
        acodec,
        "-ac",
        str(ac),
        "-ar",
        str(ar),
        "pipe:1",  # Output to pipe
    ]

    # Run FFmpeg process
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,  # Pipe input
        stdout=subprocess.PIPE,  # Pipe output
        stderr=subprocess.PIPE,  # Pipe error
    )

    # Write input audio data to the process
    ffmpeg_output, ffmpeg_error = ffmpeg_process.communicate(input=input_audio_bytes)

    # Capture the stdout audio data into a bytes-like object
    output_audio_bytes = io.BytesIO(ffmpeg_output)
    output_audio_bytes = output_audio_bytes.read()

    # save to wav file for testing purposes
    with open("webapp/testing/wav/audio.wav", "wb") as f:
        f.write(output_audio_bytes)

    audio_data, sampling_frequency = sf.read(io.BytesIO(output_audio_bytes))

    return output_audio_bytes


def run_performance_report(
    audio_bytes: bytes, sample_rate: int, sample_size: int, locale: str, n_channels: int
) -> dict:
    wav_values, sampling_frequency = sf.read(io.BytesIO(audio_bytes))

    # convert to mono (if needed)
    if n_channels > 1:
        wav_values = np.mean(wav_values, axis=1)

    snd = MySound(
        wav_values=wav_values,
        sampling_frequency=sampling_frequency,
        sample_size=sample_size,
        locale=locale,
        audio_bytes=audio_bytes,
    )
    performance_report = snd.getPerformanceReport()

    return performance_report


def run_performance_report_from_wav(wav_file_path: str, locale: str = "none") -> dict:
    # locale="none" means that we don't want to use speech recognition by default
    wav_values, sampling_frequency = sf.read(wav_file_path)
    wav_info = sf.info(wav_file_path)
    sample_size = int(wav_info.subtype[-2:])  # sample_size is in bits
    n_channels = wav_info.channels

    with open(wav_file_path, "rb") as f:
        wav_bytes = (
            f.read()
        )  # here we're assuming that the file is mono (important for later on)

    snd = MySound(
        wav_values,
        sampling_frequency,
        sample_size=sample_size,
        locale=locale,
        audio_bytes=wav_bytes,
    )
    performance_report = snd.getPerformanceReport()

    return performance_report


class MySound(pm.Sound):
    """MySound is a subclass of pm.Sound that adds a few methods to the pm.Sound class."""

    def __init__(
        self, wav_values=None, sampling_frequency=None, filename=None, **kwargs
    ):
        """Constructor for the MySound class."""
        if filename is not None:
            super().__init__(filename)
            with open(filename, "rb") as f:
                self.audio_bytes = f.read()
            info = sf.info(filename)
            self.sample_size = int(info.subtype[-2:])
        else:
            super().__init__(wav_values, sampling_frequency)
            self.sample_size = kwargs.get(
                "sample_size", None
            )  # sample_size is in bits (whereas sample_width is in bytes)
            self.audio_bytes = kwargs.get("audio_bytes", None)

        # with both constructors, we need to set the sample_rate and the locale
        self.sample_rate = int(
            self.sampling_frequency
        )  # pm.Sound already offers self.sampling_frequency but it's float and we want int
        self.locale = kwargs.get("locale", None)

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

    def getFilteredPitchInfo(self):
        """Returns the information about the *filtered* pitch of the sound as a tuple (mean, std, min, max)."""
        pitch = self.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values[pitch_values == 0] = np.nan
        pitch_values = pitch_values[~np.isnan(pitch_values)]
        # remove nan values
        pitch_values = pitch_values[~np.isnan(pitch_values)]

        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        z_scores = (pitch_values - pitch_mean) / pitch_std
        outlier_threshold = 3
        outlier_mask = np.abs(z_scores) > outlier_threshold
        filtered_pitch_values = pitch_values[~outlier_mask]

        filtered_pitch_mean = np.mean(filtered_pitch_values)
        filtered_pitch_std = np.std(filtered_pitch_values)
        filtered_pitch_min = np.min(filtered_pitch_values)
        filtered_pitch_max = np.max(filtered_pitch_values)

        return (
            filtered_pitch_mean,
            filtered_pitch_std,
            filtered_pitch_min,
            filtered_pitch_max,
        )

    def getHarmonicity(self):
        """Returns the harmonicity of the sound as a numpy array."""
        harmonicity = self.to_harmonicity()
        return {"x": harmonicity.xs().tolist(), "y": harmonicity.values[0].T.tolist()}

    def getHarmonicityInfo(self):
        """Returns the information about the harmonicity of the sound as a tuple (mean, std, min, max)."""
        harmonicity = self.to_harmonicity()
        harmonicity_values = harmonicity.values[0].T
        harmonicity_values[harmonicity_values == 0] = np.nan
        # remove nan values
        harmonicity_values = harmonicity_values[~np.isnan(harmonicity_values)]

        # remove min value
        min_harmonicity = np.min(harmonicity_values)
        harmonicity_values = harmonicity_values[harmonicity_values != min_harmonicity]

        harmonicity_mean = np.mean(harmonicity_values)
        harmonicity_std = np.std(harmonicity_values)
        harmonicity_min = np.min(harmonicity_values)
        harmonicity_max = np.max(harmonicity_values)

        return harmonicity_mean, harmonicity_std, harmonicity_min, harmonicity_max

    def getSpeechOnly(self) -> "MySound":
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
            waveform.values.T, sampling_frequency=self.sample_rate
        )

        return extracted_sound

    def getTranscription(self):
        recognizer = sr.Recognizer()
        sample_width = int(self.sample_size / 8)
        # because sample_size is in bits and sample_width is in bytes

        audio_data = sr.AudioData(self.audio_bytes, int(self.sample_rate), sample_width)
        try:
            transcript = recognizer.recognize_google(audio_data, language=self.locale)
            myprint(f'Transcript: "{transcript}"', "blue")
            words = word_tokenize(transcript)
            return transcript, words
        except sr.UnknownValueError:
            myprint(
                'Speech Recognition could not understand audio, returning transcript="" and words=[].',
                "red",
            )
            return "", []

    def getPerformanceReport(self):
        """Returns a dictionary containing the performance report of the sound."""
        myprint("Generating performance report...", "yellow")

        # get duration
        duration = self.duration

        # get transcript
        if self.locale == "none":
            myprint("No locale specified, skipping speech recognition...", "yellow")
            transcript = ""
            words_per_minute = 0
        else:
            myprint(
                f"Performing speech recognition with locale {self.locale}...", "blue"
            )
            transcript, words = self.getTranscription()
            words_per_minute = len(words) / (duration / 60)

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
            "transcript": transcript,
            "words_per_minute": words_per_minute,
        }
        overall_score = self.getOverallScore(report)
        report["overall_score"] = overall_score

        myprint("Performance report generated!", "green")
        return report

    def getOverallScore(self, report):
        """Computes the overall score."""

        # TODO: improve ffs!
        # problem: score depends on whether we're using a transcript (using a transcript will always improve the score)

        # pe stands for "percentage error"
        terms = {}

        if (
            report["words_per_minute"] != 0
        ):  # this term exists only if there is a transcript
            terms["wpm_pe"] = (report["words_per_minute"] - IDEAL_WPM) / IDEAL_WPM
        terms["pitch_std_pe"] = (
            report["pitch_std"] - IDEAL_PITCH_STD
        ) / IDEAL_PITCH_STD
        terms["intensity_std_pe"] = (
            report["intensity_std"] - IDEAL_INTENSITY_STD
        ) / IDEAL_INTENSITY_STD
        terms["ratio_speech_time_pe"] = (
            report["ratio_speech_time"] - IDEAL_RATIO_SPEECH_TIME
        ) / IDEAL_RATIO_SPEECH_TIME

        s = 0
        for value in terms.values():
            s += abs(value)
        overall_score = 1 / s

        overall_score *= 1000  # bigger is better

        return overall_score

    def plot_all_together(self):
        waveform = self.getWaveform()
        plt.plot(
            waveform["x"], waveform["y"], color="green", label="waveform", alpha=0.25
        )
        plt.yticks([])
        plt.twinx()

        intensity = self.getIntensity()
        plt.plot(intensity["x"], intensity["y"], color="red", label="intensity")
        plt.yticks([])
        plt.twinx()

        pitch = self.getPitch()
        plt.plot(pitch["x"], pitch["y"], color="green", label="pitch", marker=".")
        plt.yticks([])
        plt.twinx()

        harmonicity = self.getHarmonicity()
        min_harmonicity = np.min(harmonicity["y"])
        # only plot harmonicity where it's not min_harmonicity
        harmonicity["y"] = [
            h if h != min_harmonicity else np.nan for h in harmonicity["y"]
        ]
        plt.plot(harmonicity["x"], harmonicity["y"], color="blue", label="harmonicity")
        plt.yticks([])

        # add custom legend: red line for intensity, green line for pitch, blue line for harmonicity
        intensity_patch = plt.Line2D([0], [0], color="red", label="intensity")
        pitch_patch = plt.Line2D([0], [0], color="green", label="pitch", marker=".")
        harmonicity_patch = plt.Line2D([0], [0], color="blue", label="harmonicity")
        plt.legend(handles=[intensity_patch, pitch_patch, harmonicity_patch])

        plt.show()

    def plot_all_separate(self):
        waveform = self.getWaveform()
        # create 3 subplots vertically stacked
        fig, axs = plt.subplots(3, 1, figsize=(20, 20))

        # first subplot: waveform
        ax = axs[0]
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.plot(
            waveform["x"], waveform["y"], color="green", label="waveform", alpha=0.25
        )
        ax = ax.twinx()
        ax.set_ylabel("Intensity (dB)")
        intensity = self.getIntensity()
        intensity_std = np.std(intensity["y"])
        intensity_mean = np.mean(intensity["y"])
        ax.plot(intensity["x"], intensity["y"], color="red", label="intensity")
        ax.set_title(
            f"Intensity std: {intensity_std:.2f}; Intensity mean: {intensity_mean:.2f}"
        )

        # second subplot: pitch
        ax = axs[1]
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.plot(
            waveform["x"], waveform["y"], color="green", label="waveform", alpha=0.25
        )
        ax = ax.twinx()
        ax.set_ylabel("Pitch (Hz)")
        pitch = self.getPitch()
        df_pitch = pd.DataFrame({"x": pitch["x"], "y": pitch["y"]})
        df_pitch.dropna(inplace=True)  # remove nan values
        pitch_std = df_pitch["y"].std()
        pitch_mean = df_pitch["y"].mean()
        z_scores = (df_pitch["y"] - pitch_mean) / pitch_std
        outlier_threshold = 3
        outlier_mask = np.abs(z_scores) > outlier_threshold

        df_pitch_filtered = df_pitch[~outlier_mask]
        outliers = df_pitch[outlier_mask]
        no_outlier_pitch_std = df_pitch_filtered["y"].std()
        ax.plot(
            df_pitch_filtered["x"],
            df_pitch_filtered["y"],
            color="green",
            label="pitch (outliers removed)",
            marker=".",
        )
        ax.plot(
            outliers["x"],
            outliers["y"],
            color="red",
            label="outliers",
            marker="o",
            linestyle="",
        )
        ax.set_title(
            f"Pitch std: {pitch_std:.2f}. No outlier pitch std: {no_outlier_pitch_std:.2f}"
        )

        # third subplot: harmonicity
        ax = axs[2]
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time (s)")
        ax.plot(
            waveform["x"], waveform["y"], color="green", label="waveform", alpha=0.25
        )
        ax = ax.twinx()
        ax.set_ylabel("Harmonicity")
        harmonicity = self.getHarmonicity()
        min_harmonicity = np.min(harmonicity["y"])
        # only plot harmonicity where it's not min_harmonicity
        harmonicity["y"] = [
            h if h != min_harmonicity else np.nan for h in harmonicity["y"]
        ]
        harmonicity_std = np.nanstd(harmonicity["y"])  # ignore nan values
        harmonicity_mean = np.nanmean(harmonicity["y"])  # ignore nan values
        # add a horizontal line at the mean harmonicity
        ax.axhline(
            harmonicity_mean,
            color="blue",
            label=f"mean harmonicity",
            linestyle="--",
        )
        ax.plot(harmonicity["x"], harmonicity["y"], color="blue", label="harmonicity")
        ax.set_title(
            f"Harmonicity std: {harmonicity_std:.2f}. Harmonicity mean: {harmonicity_mean:.2f}"
        )

        plt.show()

    def get_features(self):
        """Returns relevant features of the sound as a dictionary."""
        intensity_mean, intensity_std, intensity_max = self.getIntensityInfo()
        pitch_mean, pitch_std, pitch_min, pitch_max = self.getFilteredPitchInfo()
        (
            harmonicity_mean,
            harmonicity_std,
            harmonicity_min,
            harmonicity_max,
        ) = self.getHarmonicityInfo()

        features = {
            "intensity_std": intensity_std,
            "pitch_std": pitch_std,
            "harmonicity_mean": harmonicity_mean,
        }

        return features

    def propose_score(self):
        """Proposes a score for the sound."""
        intensity_mean, intensity_std, intensity_max = self.getIntensityInfo()
        pitch_mean, pitch_std, pitch_min, pitch_max = self.getFilteredPitchInfo()
        (
            harmonicity_mean,
            harmonicity_std,
            harmonicity_min,
            harmonicity_max,
        ) = self.getHarmonicityInfo()

        print(f"Intensity std: {intensity_std:.2f}")
        print(f"Pitch std: {pitch_std:.2f}")
        print(f"Harmonicity mean: {harmonicity_mean:.2f}")


def myprint(text, color="white"):
    text = str(text)  # in case we pass a number
    if color == "red":
        print("\033[91m" + text + "\033[0m")
    elif color == "green":
        print("\033[92m" + text + "\033[0m")
    elif color == "yellow":
        print("\033[93m" + text + "\033[0m")
    elif color == "blue":
        print("\033[94m" + text + "\033[0m")
    elif color == "magenta":
        print("\033[95m" + text + "\033[0m")
    elif color == "cyan":
        print("\033[96m" + text + "\033[0m")
    else:  # if color == "white" or any other value for color
        print(text)
