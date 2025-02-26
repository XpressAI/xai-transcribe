from xai_components.base import InArg, OutArg, InCompArg, Component, xai_component, dynalist, dynatuple
import torch

@xai_component
class LoadAudioFile(Component):
    """
    Loads an audio file for processing.

    ##### inPorts:
    - file_path (str): Path to the audio file to be loaded.
    
    ##### outPorts:
    - audio_data (dict): Dictionary containing the audio array and sampling rate.
    """
    file_path: InCompArg[str]
    audio_data: OutArg[dict]

    def execute(self, ctx) -> None:
        from datasets import load_dataset
        import numpy as np
        
        file_path = self.file_path.value
        
        try:
            # Load from file path
            import soundfile as sf
            audio_array, sampling_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
                
            audio_dict = {
                "array": audio_array,
                "sampling_rate": sampling_rate
            }
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            print("Loading sample audio from LibriSpeech dataset instead...")
            
            # Fallback to sample dataset
            concatenated_librispeech = load_dataset(
                "sanchit-gandhi/concatenated_librispeech", split="train", streaming=True
            )
            sample = next(iter(concatenated_librispeech))
            audio_dict = sample["audio"]
            
        self.audio_data.value = audio_dict
        print(f"Loaded audio with sampling rate: {audio_dict['sampling_rate']}Hz, duration: {len(audio_dict['array'])/audio_dict['sampling_rate']:.2f}s")


@xai_component
class PlayAudio(Component):
    """
    Provides information about the loaded audio file.
    (Note: Audio playback is not available outside of Jupyter notebooks)

    ##### inPorts:
    - audio_data (dict): Dictionary containing the audio array and sampling rate.
    """
    audio_data: InCompArg[dict]

    def execute(self, ctx) -> None:
        audio_dict = self.audio_data.value
        
        if audio_dict:
            duration = len(audio_dict["array"]) / audio_dict["sampling_rate"]
            print(f"Audio information: {duration:.2f} seconds, {audio_dict['sampling_rate']}Hz")
            print("Note: Audio playback is only available in Jupyter notebook environments")
        else:
            print("No audio data available.")


@xai_component
class SpeakerDiarization(Component):
    """
    Performs speaker diarization to identify "who spoke when" in an audio file.

    ##### inPorts:
    - audio_data (dict): Dictionary containing the audio array and sampling rate.
    - use_auth_token (bool): Whether to use authentication token for accessing the model.
    
    ##### outPorts:
    - diarization_result (list): List of speaker segments with start/end times.
    """
    audio_data: InCompArg[dict]
    use_auth_token: InCompArg[bool]
    diarization_result: OutArg[list]

    def __init__(self):
        super().__init__()
        self.use_auth_token.value = True

    def execute(self, ctx) -> None:
        from pyannote.audio import Pipeline
        
        audio_dict = self.audio_data.value
        use_auth_token = self.use_auth_token.value
        
        print("Loading speaker diarization pipeline...")
        try:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1", 
                use_auth_token=use_auth_token
            )
            
            # Convert audio to tensor format expected by pyannote
            input_tensor = torch.from_numpy(audio_dict["array"][None, :]).float()
            
            print("Running speaker diarization...")
            outputs = diarization_pipeline(
                {"waveform": input_tensor, "sample_rate": audio_dict["sampling_rate"]}
            )
            
            # Extract the diarization results
            diarization_segments = outputs.for_json()["content"]
            
            self.diarization_result.value = diarization_segments
            
            # Print a summary of the diarization
            print(f"Found {len(diarization_segments)} speaker segments:")
            for segment in diarization_segments:
                speaker = segment["label"]
                start = segment["segment"]["start"]
                end = segment["segment"]["end"]
                print(f"  {speaker}: {start:.2f}s - {end:.2f}s")
                
        except Exception as e:
            print(f"Error in speaker diarization: {e}")
            print("Make sure you have accepted the terms of use for the models at:")
            print("- https://huggingface.co/pyannote/speaker-diarization")
            print("- https://huggingface.co/pyannote/segmentation")
            self.diarization_result.value = []


@xai_component
class SpeechTranscription(Component):
    """
    Transcribes speech in an audio file using Whisper model.

    ##### inPorts:
    - audio_data (dict): Dictionary containing the audio array and sampling rate.
    - model_name (str): Name of the Whisper model to use.
    
    ##### outPorts:
    - transcription_result (dict): Dictionary containing the transcription with timestamps.
    """
    audio_data: InCompArg[dict]
    model_name: InCompArg[str]
    transcription_result: OutArg[dict]

    def __init__(self):
        super().__init__()
        self.model_name.value = "openai/whisper-base"

    def execute(self, ctx) -> None:
        from transformers import pipeline
        
        audio_dict = self.audio_data.value
        model_name = self.model_name.value
        
        print(f"Loading ASR pipeline with model: {model_name}")
        try:
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
            )
            
            print("Transcribing audio...")
            result = asr_pipeline(
                audio_dict.copy(),
                generate_kwargs={"max_new_tokens": 256},
                return_timestamps=True,
            )
            
            self.transcription_result.value = result
            
            # Print a summary of the transcription
            print(f"Transcription complete with {len(result['chunks'])} segments")
            print(f"Full text: {result['text'][:100]}...")
            
        except Exception as e:
            print(f"Error in speech transcription: {e}")
            self.transcription_result.value = None


@xai_component
class CombineDiarizationAndTranscription(Component):
    """
    Combines speaker diarization and transcription results to create a meeting transcript.

    ##### inPorts:
    - diarization_result (list): List of speaker segments with start/end times.
    - transcription_result (dict): Dictionary containing the transcription with timestamps.
    
    ##### outPorts:
    - combined_transcript (list): List of segments with speaker, text, and timestamps.
    """
    diarization_result: InCompArg[list]
    transcription_result: InCompArg[dict]
    combined_transcript: OutArg[list]

    def execute(self, ctx) -> None:
        try:
            from speechbox import ASRDiarizationPipeline
            
            diarization = self.diarization_result.value
            transcription = self.transcription_result.value
            
            if not diarization or not transcription:
                print("Missing diarization or transcription results")
                self.combined_transcript.value = []
                return
                
            # Manually combine the results
            # This is a simplified version of what the ASRDiarizationPipeline does
            combined_results = []
            
            # Create a mapping of transcription chunks to their timestamps
            transcription_chunks = transcription["chunks"]
            
            # For each speaker segment, find the overlapping transcription chunks
            for speaker_segment in diarization:
                speaker = speaker_segment["label"]
                speaker_start = speaker_segment["segment"]["start"]
                speaker_end = speaker_segment["segment"]["end"]
                
                # Find all transcription chunks that overlap with this speaker segment
                segment_text = []
                segment_start = None
                segment_end = None
                
                for chunk in transcription_chunks:
                    chunk_start, chunk_end = chunk["timestamp"]
                    
                    # Check if this chunk overlaps with the speaker segment
                    if (chunk_start <= speaker_end and chunk_end >= speaker_start):
                        segment_text.append(chunk["text"])
                        
                        # Update segment start/end times
                        if segment_start is None or chunk_start > segment_start:
                            segment_start = chunk_start
                        if segment_end is None or chunk_end > segment_end:
                            segment_end = chunk_end
                
                if segment_text:
                    combined_results.append({
                        "speaker": speaker,
                        "text": " ".join(segment_text),
                        "timestamp": (segment_start, segment_end)
                    })
            
            self.combined_transcript.value = combined_results
            
            # Print the formatted transcript
            print("Combined transcript:")
            formatted_transcript = self.format_as_transcription(combined_results)
            print(formatted_transcript)
            
        except Exception as e:
            print(f"Error combining results: {e}")
            print("If you're missing the speechbox package, install it with:")
            print("pip install git+https://github.com/huggingface/speechbox")
            self.combined_transcript.value = []
    
    def tuple_to_string(self, start_end_tuple, ndigits=1):
        return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))
    
    def format_as_transcription(self, raw_segments):
        return "\n\n".join(
            [
                chunk["speaker"] + " " + self.tuple_to_string(chunk["timestamp"]) + chunk["text"]
                for chunk in raw_segments
            ]
        )


@xai_component
class SaveTranscriptToFile(Component):
    """
    Saves the meeting transcript to a file.

    ##### inPorts:
    - combined_transcript (list): List of segments with speaker, text, and timestamps.
    - output_file (str): Path to save the transcript file.
    """
    combined_transcript: InCompArg[list]
    output_file: InCompArg[str]

    def __init__(self):
        super().__init__()
        self.output_file.value = "meeting_transcript.txt"

    def execute(self, ctx) -> None:
        transcript = self.combined_transcript.value
        output_file = self.output_file.value
        
        if not transcript:
            print("No transcript data to save")
            return
            
        try:
            def tuple_to_string(start_end_tuple, ndigits=1):
                return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))
            
            with open(output_file, "w") as f:
                f.write("MEETING TRANSCRIPT\n")
                f.write("=================\n\n")
                
                for segment in transcript:
                    speaker = segment["speaker"]
                    text = segment["text"]
                    timestamp = tuple_to_string(segment["timestamp"])
                    
                    f.write(f"{speaker} {timestamp}: {text}\n\n")
            
            print(f"Transcript saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving transcript: {e}")
