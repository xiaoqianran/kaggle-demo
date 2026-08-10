# Audio From Scratch · Progress

**Kaggle:** [xiaosuhuaer/grok-audio-from-scratch](https://www.kaggle.com/code/xiaosuhuaer/grok-audio-from-scratch) · COMPLETE · T4×2  
**Acceptance:** `PASS` · `results/audio_from_scratch/ACCEPTANCE.json`  
**failures:** []

| Stage | Status | Metric |
|-------|--------|--------|
| S00 | ✅ | STFT/Mel/MFCC |
| S01 | ✅ | segments=3 |
| S02 | ✅ | hand=0.975 cnn=1.0 |
| S03 | ✅ | DTW acc=0.975 |
| S04 | ✅ | CTC exact=7 demos=[['a', 'a'], ['b', 'b'], ['c', 'c'], ['ab', 'ab'], ['abc', 'abc'], ['cab', 'cab'], ['bca', 'bca']] |
| S05 | ✅ | soft=1.0 n_good=3 |
| S06 | ✅ | formant TTS |
| S07 | ✅ | spec_conv=0.18951663374900818 |
| S08 | ✅ | 4 scenes TTA |
| S09 | ✅ | sine F0 160.0→202.53164556962025 (ratio=1.2658, expect≈1.2599) |
| S10 | ✅ | semantic_ok=True ASR=['hello world', 'open the door'] replies=['hello world', 'yes'] |

## Bugfix (this session)
1. **S09 pitch inverted / phase-vocoder broken on pure tones** → switched to **resampling pitch shift**; sine probe ratio≈1.266
2. Hard acceptance gates include S09 sine F0 direction + ratio
3. Re-verified full S00–S10 on Kaggle after fix

## Capability ladder
```
waveform → features → VAD → classify → DTW → CTC → Whisper
                     ↘ formant TTS → Griffin-Lim → TTA → A2A → agent
```
