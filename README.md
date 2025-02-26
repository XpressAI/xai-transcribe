# Xircuits Audio Transcription Components

<img width="1554" alt="image" src="https://github.com/user-attachments/assets/608c3c2d-168d-47c9-b623-5f8dc0bc15bc" />

A Xircuits component library for transcribing audio into text with speaker diarization. This library provides components for:

- Loading and processing audio files
- Performing speaker diarization (identifying who spoke when)
- Transcribing speech to text
- Combining diarization and transcription results
- Saving formatted transcripts

## Prerequisites

This component library requires:

1. Access to the Hugging Face Hub models:
   - You need to accept the terms of use for the pyannote models at:
     - https://huggingface.co/pyannote/speaker-diarization
     - https://huggingface.co/pyannote/segmentation
   - A Hugging Face access token for the diarization models

2. Sufficient disk space for the downloaded models (approximately 1-2GB)

## Installation

To use this component library, ensure you have Xircuits installed, then simply run:

```
xircuits install https://github.com/xpressai/xai-transcribe
```

Alternatively you may manually copy the directory / clone or submodule the repository to your working Xircuits project directory then install the packages using:

```
pip install -r requirements.txt
```

## Usage

The library provides components for a complete audio transcription pipeline:

1. `TranscribeLoadAudioFile` - Load an audio file or use a sample dataset
2. `TranscribeSpeakerDiarization` - Identify different speakers in the audio
3. `TranscribeSpeechTranscription` - Transcribe the audio to text with timestamps
4. `TranscribeCombineDiarizationAndTranscription` - Combine speaker information with transcription
5. `TranscribeSaveTranscriptToFile` - Save the formatted transcript to a file

## Example

Create a new Xircuits workflow and add the components in sequence:
1. Start with `TranscribeLoadAudioFile` and provide a path to your audio file
2. Connect to `TranscribeSpeakerDiarization` (set use_auth_token to True if using Hugging Face models)
3. Add `TranscribeSpeechTranscription` (defaults to Whisper base model)
4. Connect both to `TranscribeCombineDiarizationAndTranscription`
5. Finally connect to `TranscribeSaveTranscriptToFile` to save the results

## Tests
A github action to test your workflow runs has been provided. Simply add the path of your workflows [here](.github/workflows/run-workflow-tests.yml#L11).
