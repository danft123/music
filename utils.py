import pandas
import numpy as np
import soundfile as sf
import librosa

#Extract segments from csv instructions
def read_data(annot, date_format):
  df = pandas.read_csv(annot, sep=',')

  # Use proper pandas datatypes
  df['Time'] = pandas.to_datetime(df['DateTimeStamp'], format=date_format)
  df['Duration'] = pandas.to_timedelta(df['Duration ms'], unit='ms')
  df = df.drop(columns=['DateTimeStamp', 'Duration ms'])

  # Compute start and end time of each segment
  # audio starts at time of first segment
  first = df['Time'].iloc[0]
  df['Start'] = df['Time'] - first
  df['End'] = df['Start'] + df['Duration']

  return df


def extract_segments(y, sr, segments):
  # compute segment regions in number of samples
  starts = np.floor(segments.Start.dt.total_seconds() * sr).astype(int)
  ends = np.ceil(segments.End.dt.total_seconds() * sr).astype(int)

  # slice the audio into segments
  i = 0
  for start, end in zip(starts, ends):
    audio_seg = y[start:end]
    print('extracting audio segment:', len(audio_seg), 'samples')

    # file name string
    # it takes 5 first character of Action
    # and converts start and end time
    file_name = str(segments.Action[i][:5]) + \
        '__' + \
        str(segments.Start[i]).split('s ')[1].replace(':', '_') + \
        '__' + \
        str(segments.End[i]).split('s ')[1].replace(':', '_') + ".wav"

    sf.write(file_name, audio_seg, sr)
    i += 1


#decomposer
def decomposer(wav_file):
    y,sr = librosa.load(wav_file)
    mel = librosa.feature.mfcc(y=y,sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return {"y":y,"sr":sr,"mel":mel,"cent":cent,"zero_crossing":zero_crossing,"chroma":chroma,"rolloff":rolloff}
