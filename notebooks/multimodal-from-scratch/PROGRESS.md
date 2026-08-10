# Progress · Multimodal From Scratch

> 全部 **FS00–FS14** 已在 Kaggle **T4×2** 跑通并验收。

| Stage | Method | Key metrics | vs prev |
|-------|--------|-------------|---------|
| FS00 |  | `{}` |  |
| FS01 | bag-of-colors + geometry heuristics | `{"exact_match_acc": 0.6666666666666666}` | FS00 only scored pairs; FS01 emits captions |
| FS02 | TinyCNN classifier to class name string | `{"val_acc": 1.0, "final_hist": {"epoch": 7, "train_loss": 0.0379, "val_acc": 1.0` | FS01 fixed rules; FS02 learns filters (closed-set  |
| FS03 | Show-and-Tell CNN encoder + LSTM decoder | `{"final_hist": {"epoch": 11, "loss": 0.5091, "val_token_acc": 0.8203}}` | FS02 closed-set class word; FS03 open sequence gen |
| FS04 | Show-Attend-Tell soft attention over spa | `{"final_hist": {"epoch": 11, "loss": 0.4773, "val_token_acc": 0.8516}}` | FS03 single global vector; FS04 looks at different |
| FS05 | dual-encoder InfoNCE (CLIP mini) | `{"R@1": 1.0, "final_hist": {"epoch": 20, "loss": 2.8004, "batch_acc": 0.0625}}` | FS03/04 generate text; FS05 learns shared embeddin |
| FS06 | CNN+question-emb concat fusion VQA | `{"acc": 1.0, "final_hist": {"epoch": 15, "loss": 0.0005, "acc": 1.0}}` | FS05 retrieves similar images; FS06 answers free q |
| FS07 | page render + OCR text + TF-IDF retrieve | `{"retrieval_acc": 1.0, "answer_acc": 1.0}` | FS06 answers about shapes; FS07 answers from multi |
| FS08 | patch embeddings + MaxSim (ColPali-style | `{"visual_R@1": 0.6, "text_tfidf_R@1": 1.0, "final_hist": {"epoch": 25, "loss": 1` | FS07 retrieves on OCR text only; FS08 retrieves fr |
| FS09 | frame CNN + GRU temporal pool -> video l | `{"clean_acc": 0.6666666666666666, "final_hist": {"epoch": 24, "loss": 0.0006, "t` | FS02/03 single image; FS09 models motion over time |
| FS10 | log-mel + CNN/BiGRU utterance ASR (synth | `{"acc": 0.75, "final_hist": {"epoch": 29, "loss": 0.4743, "val_acc": 0.7}}` | FS09 video frames; FS10 time-frequency audio pathw |
| FS11 | class-conditioned mini DDPM image genera | `{"color_l2_to_gt_mean": {"red_circle": 0.33716055750846863, "blue_square": 0.348` | earlier stages understand/describe images; FS11 sy |
| FS12 | conditional latent MLP video generator ( | `{"frame_mse": {"red_circle": 3.83851556762238e-06, "blue_square": 4.084728061570` | FS11 single image; FS12 emits temporal sequence co |
| FS13 | tiny multimodal LM: image tokens + text  | `{"acc": 1.0, "final_hist": {"epoch": 49, "loss": 0.0024, "token_acc": 0.9992}}` | FS06 separate fusion head; FS13 single autoregress |
| FS14 | any-to-any router with shared trunk + mo | `{"route_acc": 0.8333333333333334, "final_hist": {"epoch": 59, "loss": 5.6928, "l` | FS13 single VQA LM interface; FS14 multi-route in_ |

## Kernels

- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs00-fs02-foundations
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs03-fs04-captioning
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs05-fs06-align-vqa
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs07-fs08-doc
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs09-fs10-video-audio
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs11-fs12-generation
- https://www.kaggle.com/code/qiaojiajin/grok-multimodal-fs13-fs14-unified

## Task coverage

| Task | Stages |
|------|--------|
| 图像→文本 | FS01–FS05 |
| 视觉问答 | FS06, FS13 |
| 文档问答 | FS07 |
| 可视化文档检索 | FS08 |
| 视频→文本 | FS09 |
| 音频→文本 | FS10 |
| 图文→图 | FS11, FS14 |
| 图文→视频 | FS12 |
| Any-to-Any | FS14 |

Last update: full curriculum COMPLETE
