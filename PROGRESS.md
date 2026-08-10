# Audio From Scratch · Progress

**Kaggle run:** [xiaosuhuaer/grok-audio-from-scratch](https://www.kaggle.com/code/xiaosuhuaer/grok-audio-from-scratch) · `KernelWorkerStatus.COMPLETE`  
**Accelerator:** NvidiaTeslaT4 (T4×2) · **Artifacts:** `results/audio_from_scratch/`

| Stage | Status | Highlight |
|-------|--------|-----------|
| S00 Foundations STFT/Mel/MFCC | ✅ | waveform→STFT→Mel→MFCC feature chain |
| S01 VAD energy+flatness | ✅ | energy vs spectral flatness vs AND |
| S02 Audio classification | ✅ | handcrafted centroid vs Mel-CNN (both ~1.0 on synthetic) |
| S03 ASR DTW templates | ✅ | MFCC sequence + DTW isolated digits |
| S04 ASR CTC toy | ✅ | BiLSTM-CTC on {a,b,c} with blank collapse |
| S05 Whisper-tiny | ✅ | espeak English → natural language ASR |
| S06 Formant TTS | ✅ | text→phones→f0+formants→wave |
| S07 Griffin-Lim vocoder | ✅ | magnitude STFT phase reconstruction |
| S08 Text→Audio scenes | ✅ | procedural rain/alarm/footsteps/wind |
| S09 Audio→Audio FX | ✅ | pitch/time/denoise/timbre (F0↑ ~1.26×) |
| S10 Full voice pipeline | ✅ | merged VAD + Whisper: `hello world`/`open the door` → replies |

## Capability ladder

```
waveform → features → VAD → classify → align(DTW) → CTC → Whisper
                                 ↘ formant TTS → vocoder → TTA → A2A → agent loop
```

## S10 key lesson

Fine-grained VAD chops destroy ASR (`you` / `that i will`).  
Phrase-level merged segments restore Whisper: **`hello world`**, **`open the door`** → policy replies **`hello world`**, **`yes`**.

## Map

See [docs/audio-from-scratch/ROADMAP.md](docs/audio-from-scratch/ROADMAP.md).
