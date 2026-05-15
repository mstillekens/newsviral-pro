# Voice samples

This directory holds the **reference audio** that gets used to train the
narrator voice for the newsroom.

## To enable a chilango (Mexico City) accent

1. Get a 30–60 second WAV/MP3 of clear Mexican Spanish speech.
   - **Best**: record yourself or someone you know reading a paragraph from a
     Mexican newspaper. Read naturally — not fast, not theatrical.
   - **Alternative**: pull a Creative-Commons-licensed clip of a Mexican
     Spanish speaker from Mozilla Common Voice
     (commonvoice.mozilla.org/es/datasets), filtered to the `mx_locale`.
   - Acoustic quality matters more than mic quality: a quiet room and a
     phone-quality recording usually beats a fancy mic in a noisy room.

2. Save the file here, e.g. `voice_samples/chilango.wav`.

3. Train:

   ```bash
   python clone_voice.py voice_samples/chilango.wav
   ```

   The script uploads the sample to MiniMax via Replicate, returns a
   `voice_id`, and writes `MINIMAX_VOICE_ID=…` into `.env`. The next pipeline
   run will use that voice automatically (no code changes needed).

4. Verify the new voice with a single test video:

   ```bash
   python news_viral_pro.py --auto 1 --max 4
   ```

## Constraints (from MiniMax)

- Format: MP3, M4A, or WAV.
- Duration: 10 s minimum, 5 min maximum.
- Size: under 20 MB.
- Single speaker, no music, minimal background noise.

## What "good" sounds like

- Clean diction without slurring.
- Natural pace (about 140–160 words per minute).
- Includes a range of tones (a few questions or emphatic moments are great).
- Recorded in mono or stereo at 22 kHz+ sample rate.

## Reverting to the default voice

Remove (or comment out) `MINIMAX_VOICE_ID` from `.env`. The pipeline falls
back to `English_Wiselady` with `language_boost=Spanish`.
