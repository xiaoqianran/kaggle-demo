# Audio From Scratch · Progress

**Kaggle:** [xiaosuhuaer/grok-audio-from-scratch](https://www.kaggle.com/code/xiaosuhuaer/grok-audio-from-scratch) · COMPLETE · T4×2  
**Acceptance:** `PASS` · file `results/audio_from_scratch/ACCEPTANCE.json`

| Stage | Status | Metric |
|-------|--------|--------|
| S00 Foundations | ✅ | STFT/Mel/MFCC from scratch |
| S01 VAD | ✅ | n_segments=3 |
| S02 Classification | ✅ | hand=0.975 cnn=1.0 |
| S03 ASR DTW | ✅ | acc=0.975 |
| S04 ASR CTC | ✅ | demos=[['a', 'a'], ['b', 'b'], ['c', 'c'], ['ab', 'ab'], ['abc', 'abc'], ['cab', 'cab'], ['bca', 'bca']] exact=7 |
| S05 Whisper | ✅ | soft=1.0 n_good=3 |
| S06 Formant TTS | ✅ | lexicon TTS wavs |
| S07 Griffin-Lim | ✅ | spec_conv=0.18951663374900818 |
| S08 Text→Audio | ✅ | 4 scenes |
| S09 Audio→Audio | ✅ | F0 133.33333333333334→168.42105263157896 |
| S10 Pipeline | ✅ | semantic_ok=True transcripts=['hello world', 'open the door'] replies=['hello world', 'yes'] |

## Capability ladder
```
waveform → features → VAD → classify → DTW → CTC → Whisper
                         ↘ formant TTS → Griffin-Lim → TTA → A2A → agent
```

## Bugfix log (this pass)
- S01: louder speech + looser thresholds → multi-segment VAD
- S03: distinct formants + CMVN/deltas + band DTW → acc 0.975
- S04: curriculum + stronger model → full demo exact match a/b/c/ab/abc/cab/bca
- S05: clearer espeak + beam decode → soft_match 1.0 on 3/3
- Added hard ACCEPTANCE gates so silent regressions fail the kernel

Map: `docs/audio-from-scratch/ROADMAP.md`
