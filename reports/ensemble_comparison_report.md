# Automated Ensemble Optimization Report

Generated at: Mon Jan 12 05:17:22 2026

## Winning Configuration
**Models**: Ateeqq/ai-vs-human-image-detector, haywoodsloan/ai-image-detector-dev-deploy, Organika/sdxl-detector

### Slice-Wise Performance & Weights
| Slice | BA | Margin | FP | FN | w1 | w2 | w3 |
|-------|----|--------|----|----|----|----|----|
| >500k | 0.7711 | 1.98 | 258 | 2168 | 0.192 | 0.795 | 0.013 |
| 50k-500k | 0.7990 | -1.72 | 435 | 37 | 0.423 | 0.334 | 0.243 |
| 10k-50k | 1.0000 | 1.34 | 0 | 0 | 0.210 | 0.561 | 0.229 |

## Model Contribution Analysis
- **Ateeqq/ai-vs-human-image-detector**: Average weight contribution: 0.28
- **haywoodsloan/ai-image-detector-dev-deploy**: Average weight contribution: 0.56
- **Organika/sdxl-detector**: Average weight contribution: 0.16

## Suspicion Window Tracking
Found 42 images in the suspicion window (|logit - margin| < 0.1123857305657932):

- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_6560.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/parvesh_ai_884.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5325.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_4738.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5555.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5554.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_6656.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_3063.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5604.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_6251.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_4321.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5428.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_4750.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_4419.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_5510.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_5086.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_4988.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_3554.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_3887.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/ai/prithiv_ai_deepfake_4192.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_1326.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_1680.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_2835.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_1395.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_4038.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_1804.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/raid_real_178.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_3396.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_2164.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_3022.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_1915.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/raid_real_0.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/raid_real_126.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_2526.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_4859.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_146.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_246.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_real_2694.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_2631.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_2034.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_4554.jpg`
- `/Users/netanel.ossi/Desktop/ai_detector_dataset/real/parvesh_fill_4221.jpg`
