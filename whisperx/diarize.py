import numpy as np
import pandas as pd
from pyannote.audio import Pipeline

class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization@2.1",
        use_auth_token=None,
    ):
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)

    def __call__(self, audio, min_speakers=None, max_speakers=None):
        segments = self.model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)
        return diarize_df

def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):
    print("Assigning speakers to words...")
    for seg in result_segments:
        print(f"Seg: {seg}")
        wdf = seg['word-segments']
        if len(wdf['start'].dropna()) == 0:
            print("Assigning segment start and end times to word dataframe")
            wdf['start'] = seg['start']
            wdf['end'] = seg['end']
        speakers = []
        for wdx, wrow in wdf.iterrows():
            if not np.isnan(wrow['start']):
                print("Calculating intersection and union for word segment")
                diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
                diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
                # remove no hit
                if not fill_nearest:
                    print("Filtering out non-intersecting segments")
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                else:
                    dia_tmp = diarize_df
                if len(dia_tmp) == 0:
                    print("No speaker found for this word segment")
                    speaker = None
                else:
                    print("Assigning speaker with highest intersection")
                    speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
            else:
                print("No start time for this word segment, speaker set to None")
                speaker = None
            speakers.append(speaker)
        print("Assigning speakers to word-segments")
        seg['word-segments']['speaker'] = speakers

        print("Determining the main speaker for the segment")
        speaker_count = pd.Series(speakers).value_counts()
        print(f"Speaker count: {speaker_count}")
        if len(speaker_count) == 0:
            seg["speaker"] = "UNKNOWN"
        else:
            seg["speaker"] = speaker_count.index[0]

    print("Creating word-level segments for SRT output")
    word_seg = []
    for seg in result_segments:
        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                print("Appending word segment with speaker information")
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                word_seg.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": f"[{speaker}]: " + seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    print("Finished assigning speakers to words and creating word-level segments")

    return result_segments, word_seg


class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
