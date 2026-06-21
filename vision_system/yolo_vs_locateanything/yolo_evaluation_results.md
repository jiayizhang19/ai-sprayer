# YOLOv8 Evaluation Report

**Model Type:** yolo  
**IoU Threshold:** 0.5

## Experiment Configuration
- Model Type: **yolo**
- Device: cpu
- Dtype: float32

## Overall Results

**Total Ground Truth boxes:** 2162  
**Total Predictions:** 2159

| Metric              | Value    |
|---------------------|----------|
| True Positives (TP) | 1536 |
| False Positives (FP)| 623 |
| False Negatives (FN)| 626 |
| **Precision**       | **0.7114** |
| **Recall**          | **0.7105** |
| **F1 Score**        | **0.7109** |
| **Mean IoU**        | **0.7911** |

## Per-Class Results

A true positive requires both IoU >= threshold AND the predicted label
matching the ground truth class — so this breakdown shows which species
the model actually confuses or misses, rather than just overlapping boxes
of the wrong class.

| Class | TP | FP | FN | Precision | Recall | F1 |
|-------|----|----|----|-----------|--------|-----|
| BROST | 211 | 89 | 86 | 0.7033 | 0.7104 | 0.7069 |
| STEME | 626 | 338 | 325 | 0.6494 | 0.6583 | 0.6538 |
| URTUR | 699 | 196 | 215 | 0.7810 | 0.7648 | 0.7728 |

## Per-Image Results

| Image | GT | Pred | TP | FP | FN | F1 Score |
|-------|----|------|----|----|----|----------|
| capture_001_Brome_png.rf.009363a1240b3c2b26d5759cc46af6d5.jp... | 3 | 3 | 3 | 0 | 0 | 1.000 |
| capture_001_Brome_png.rf.87a3646cbb63b75cb3b0907ee0829db5.jp... | 3 | 3 | 3 | 0 | 0 | 1.000 |
| capture_002_Brome_png.rf.43e87e087b046a330d5471e5567ba40b.jp... | 5 | 7 | 5 | 2 | 0 | 0.833 |
| capture_002_Brome_png.rf.e8bdb86862d6c7a82c7a95146900c0b6.jp... | 5 | 6 | 5 | 1 | 0 | 0.909 |
| capture_005_Brome_png.rf.3b906de3a08aebb722ba1eb4988ad852.jp... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_006_Brome_png.rf.8052efa1a4c4b345e7e3ac6abcbf096e.jp... | 5 | 5 | 5 | 0 | 0 | 1.000 |
| capture_006_Brome_png.rf.9073473e708e05e266ff2235139ac161.jp... | 5 | 8 | 5 | 3 | 0 | 0.769 |
| capture_007_Brome_png.rf.f246600d01029cb7aca709ec311ce4d7.jp... | 5 | 6 | 4 | 2 | 1 | 0.727 |
| capture_008_Brome_png.rf.8e26d45bdcb1a8ede29714ad4b40e25f.jp... | 3 | 7 | 3 | 4 | 0 | 0.600 |
| capture_008_Brome_png.rf.d0991787486c6a55c0d11e7c99914a4d.jp... | 5 | 6 | 5 | 1 | 0 | 0.909 |
| capture_009_Brome_png.rf.5d859bbf312024da3061436ed0cbceb2.jp... | 4 | 5 | 4 | 1 | 0 | 0.889 |
| capture_011_Brome_png.rf.47ff1a01de8a0cfed076b893e203cd13.jp... | 4 | 2 | 2 | 0 | 2 | 0.667 |
| capture_011_Brome_png.rf.f10e371673c8de79a72b31471a5c2ae5.jp... | 3 | 6 | 3 | 3 | 0 | 0.667 |
| capture_014_Brome_png.rf.2557c8c6cfbdbccfadf8939438e4112d.jp... | 5 | 4 | 4 | 0 | 1 | 0.889 |
| capture_014_Brome_png.rf.e523694654e716b9958d53e258874991.jp... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_014_Brome_png.rf.fcf25bc6f133c83905064c1f49f46fe2.jp... | 5 | 7 | 5 | 2 | 0 | 0.833 |
| capture_015_Brome_png.rf.7489a48eb31b8f836f4eb9a5f19076a0.jp... | 6 | 6 | 6 | 0 | 0 | 1.000 |
| capture_015_Brome_png.rf.ce9a6fb81d4a4b70aed659f43f64040a.jp... | 6 | 6 | 6 | 0 | 0 | 1.000 |
| capture_017_Brome_png.rf.c3beddd00f78e05c3f90f6e6f0ba216d.jp... | 5 | 5 | 1 | 4 | 4 | 0.200 |
| capture_018_Brome_png.rf.0383938caf2ef56a6ad00e347caf5303.jp... | 5 | 6 | 4 | 2 | 1 | 0.727 |
| capture_020_Brome_png.rf.2343140a8a887fc568922d99a567d2ef.jp... | 5 | 9 | 3 | 6 | 2 | 0.429 |
| capture_020_Brome_png.rf.ac2b8e8df9bd3a6158d40e487499492e.jp... | 5 | 5 | 5 | 0 | 0 | 1.000 |
| capture_024_Brome_png.rf.5d520609d93965258c4a455984e60377.jp... | 3 | 5 | 3 | 2 | 0 | 0.750 |
| capture_024_Brome_png.rf.b4a5626bfdbf1350403a4bb3f22ec8a5.jp... | 5 | 7 | 4 | 3 | 1 | 0.667 |
| capture_026_Brome_png.rf.1347f0ba253afad4ee2a5e347f70f31e.jp... | 3 | 3 | 3 | 0 | 0 | 1.000 |
| capture_027_Chickweed_png.rf.0c92166b67f320b673c4f3c17fc5c4f... | 6 | 1 | 0 | 1 | 6 | 0.000 |
| capture_027_Chickweed_png.rf.4dfa626a66d6ffb55525354d88bb9ba... | 9 | 2 | 2 | 0 | 7 | 0.364 |
| capture_027_Chickweed_png.rf.764177ab7d01801da55e49a8e66e9a4... | 6 | 2 | 1 | 1 | 5 | 0.250 |
| capture_027_Chickweed_png.rf.7802cc2737c1c543569d9108ba58c99... | 9 | 4 | 2 | 2 | 7 | 0.308 |
| capture_027_Chickweed_png.rf.f29ef338c0ae69a381dd68e966258f8... | 6 | 2 | 1 | 1 | 5 | 0.250 |
| capture_027_Chickweed_png.rf.f961cf34d6adf417402e53693677d34... | 9 | 3 | 1 | 2 | 8 | 0.167 |
| capture_028_Chickweed_png.rf.08796983874c1448395fd3ddc0b8d68... | 8 | 5 | 3 | 2 | 5 | 0.462 |
| capture_028_Chickweed_png.rf.474e9e8fe591c0f1466f9efd871fcf1... | 8 | 4 | 3 | 1 | 5 | 0.500 |
| capture_028_Chickweed_png.rf.5e5bf800c5185abd919959b742595ae... | 5 | 3 | 1 | 2 | 4 | 0.250 |
| capture_028_Chickweed_png.rf.8621c03dc14df193052246a9f97499b... | 10 | 2 | 2 | 0 | 8 | 0.333 |
| capture_028_Chickweed_png.rf.91e9a59687da7f74628645fee7e04a4... | 8 | 3 | 1 | 2 | 7 | 0.182 |
| capture_028_Chickweed_png.rf.c4c3914e8678d11f77590d12d5cf15b... | 10 | 3 | 2 | 1 | 8 | 0.308 |
| capture_028_Chickweed_png.rf.ee1778b8ee90a6909a1f5d6a65b67c5... | 10 | 5 | 3 | 2 | 7 | 0.400 |
| capture_029_Chickweed_png.rf.96979064ac60f5a47ad35079af72531... | 4 | 9 | 4 | 5 | 0 | 0.615 |
| capture_032_Chickweed_png.rf.0cae03e829d4c6b567002d603a828da... | 3 | 3 | 2 | 1 | 1 | 0.667 |
| capture_036_Chickweed_png.rf.d08b4780d709fac0dfdc1b2cd151778... | 5 | 2 | 2 | 0 | 3 | 0.571 |
| capture_039_Chickweed_png.rf.a48061ade57a01f222fd347d15cfdd1... | 3 | 1 | 1 | 0 | 2 | 0.500 |
| capture_046_Chickweed_png.rf.784aabfc91af3924aeb9f150c0bc91f... | 2 | 2 | 2 | 0 | 0 | 1.000 |
| capture_047_Chickweed_png.rf.57f4525f05ec059014c2ae5b889f396... | 3 | 3 | 2 | 1 | 1 | 0.667 |
| capture_047_Chickweed_png.rf.7a2e9f80aba8b5c7d7d2e6f61fd9b65... | 3 | 3 | 2 | 1 | 1 | 0.667 |
| capture_051_Chickweed_png.rf.44a1d904bfb811fa44ecfd1fc75fcaa... | 3 | 0 | 0 | 0 | 3 | 0.000 |
| capture_081_Nettle_png.rf.e23c94e1d953c97a1581924f080ced43.j... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_084_Nettle_png.rf.6bf0e4a795e5b372f39908d248b15c5b.j... | 6 | 5 | 4 | 1 | 2 | 0.727 |
| capture_085_Nettle_png.rf.8bc851011c241cc04e1abeb0d5c84303.j... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_087_Nettle_png.rf.b1d9df9fb99c9ab81855c68f54d0362f.j... | 5 | 5 | 5 | 0 | 0 | 1.000 |
| capture_088_Nettle_png.rf.bf11d1e8c6b035fcf3ac3be217756f8b.j... | 4 | 5 | 4 | 1 | 0 | 0.889 |
| capture_088_Nettle_png.rf.f7b3cd4e66b38935656526762f74b69a.j... | 5 | 5 | 4 | 1 | 1 | 0.800 |
| capture_090_Nettle_png.rf.5c85cec5fa404895064238507efaef3d.j... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_090_Nettle_png.rf.c6e71421bb80fb1cbfc6fa00d7d16fa7.j... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_092_Nettle_png.rf.689f53f8f62c4dc314006ef3de4c5583.j... | 4 | 6 | 4 | 2 | 0 | 0.800 |
| capture_092_Nettle_png.rf.7084612be0683c52fc4a6a122afbe721.j... | 4 | 5 | 4 | 1 | 0 | 0.889 |
| capture_094_Nettle_png.rf.4c6ca8300da781cb26f9e187aa51296c.j... | 4 | 6 | 4 | 2 | 0 | 0.800 |
| capture_095_Nettle_png.rf.e8baa8d3ba2d971c2f43d7eea6063c1d.j... | 4 | 5 | 3 | 2 | 1 | 0.667 |
| capture_097_Nettle_png.rf.89a5f068ccf3463fc83c45d5892d6c62.j... | 4 | 4 | 4 | 0 | 0 | 1.000 |
| capture_098_Nettle_png.rf.bdff460c598c435c9ec80d0f8571936c.j... | 3 | 4 | 3 | 1 | 0 | 0.857 |
| capture_099_Nettle_png.rf.614bcfa669c2a47a0831a973a213e931.j... | 6 | 6 | 4 | 2 | 2 | 0.667 |
| capture_101_Nettle_png.rf.c5f89c4fb0d7678f32b324ca70b8a02f.j... | 5 | 7 | 5 | 2 | 0 | 0.833 |
| capture_102_Nettle_png.rf.523e7f8dcf7b06d795970b485b17b251.j... | 6 | 7 | 6 | 1 | 0 | 0.923 |
| capture_102_Nettle_png.rf.888db5adf9d9873fe608148026db0b07.j... | 6 | 7 | 6 | 1 | 0 | 0.923 |
| T01_Box003_2017-05-15T15-52-53-000.jpg... | 7 | 0 | 0 | 0 | 7 | 0.000 |
| T01_Box003_2017-05-25T08-49-36-496.jpg... | 94 | 113 | 83 | 30 | 11 | 0.802 |
| T01_Box003_2017-06-16T06-25-12-881.jpg... | 27 | 7 | 5 | 2 | 22 | 0.294 |
| T01_Box010_2017-05-28T10-47-58-051.jpg... | 34 | 15 | 13 | 2 | 21 | 0.531 |
| T01_Box010_2017-06-01T09-00-04-104.jpg... | 31 | 20 | 15 | 5 | 16 | 0.588 |
| T01_Box010_2017-06-04T09-05-22-860.jpg... | 28 | 18 | 10 | 8 | 18 | 0.435 |
| T01_Box010_2017-06-15T06-23-41-694.jpg... | 19 | 17 | 12 | 5 | 7 | 0.667 |
| T01_Box010_2017-06-18T06-40-54-508.jpg... | 22 | 21 | 13 | 8 | 9 | 0.605 |
| T01_Box025_2017-05-20T05-37-20-387.jpg... | 144 | 130 | 83 | 47 | 61 | 0.606 |
| T01_Box025_2017-05-29T09-21-56-780.jpg... | 73 | 75 | 56 | 19 | 17 | 0.757 |
| T01_Box025_2017-06-07T09-07-41-196.jpg... | 48 | 55 | 35 | 20 | 13 | 0.680 |
| T01_Box025_2017-06-09T12-03-06-402.jpg... | 46 | 68 | 33 | 35 | 13 | 0.579 |
| T01_Box025_2017-06-14T06-15-43-932.jpg... | 39 | 63 | 31 | 32 | 8 | 0.608 |
| T01_Box025_2017-06-16T06-49-21-438.jpg... | 39 | 60 | 30 | 30 | 9 | 0.606 |
| T01_Box025_2017-06-18T06-58-23-680.jpg... | 37 | 68 | 29 | 39 | 8 | 0.552 |
| T01_Box032_2017-06-05T09-04-33-984.jpg... | 123 | 124 | 100 | 24 | 23 | 0.810 |
| T01_Box032_2017-06-06T09-05-17-470.jpg... | 121 | 123 | 102 | 21 | 19 | 0.836 |
| T01_Box032_2017-06-07T09-07-49-102.jpg... | 104 | 103 | 87 | 16 | 17 | 0.841 |
| T01_Box032_2017-06-16T10-38-26-105.jpg... | 71 | 75 | 63 | 12 | 8 | 0.863 |
| T01_Box032_2017-06-18T06-58-35-360.jpg... | 71 | 71 | 63 | 8 | 8 | 0.887 |
| T01_Box047_2017-05-22T12-16-32-476.jpg... | 116 | 119 | 106 | 13 | 10 | 0.902 |
| T01_Box047_2017-05-31T08-39-14-342.jpg... | 44 | 54 | 37 | 17 | 7 | 0.755 |
| T01_Box047_2017-06-02T07-14-01-577.jpg... | 36 | 51 | 28 | 23 | 8 | 0.644 |
| T01_Box047_2017-06-09T06-10-08-121.jpg... | 28 | 27 | 13 | 14 | 15 | 0.473 |
| T01_Box047_2017-06-13T13-21-10-171.jpg... | 25 | 15 | 9 | 6 | 16 | 0.450 |
| T01_Box047_2017-06-15T06-21-04-595.jpg... | 18 | 12 | 11 | 1 | 7 | 0.733 |
| T01_Box054_2017-05-15T15-46-28-000.jpg... | 12 | 6 | 3 | 3 | 9 | 0.333 |
| T01_Box054_2017-06-02T13-03-22-952.jpg... | 142 | 168 | 108 | 60 | 34 | 0.697 |
| T01_Box054_2017-06-16T11-01-59-489.jpg... | 54 | 41 | 34 | 7 | 20 | 0.716 |
| T02_Box017_2017-09-04T10-46-24-202.jpg... | 0 | 0 | 0 | 0 | 0 | 0.000 |
| T02_Box017_2017-09-14T08-17-51-696.jpg... | 12 | 2 | 1 | 1 | 11 | 0.143 |
| T02_Box017_2017-09-22T08-53-50-090.jpg... | 18 | 21 | 17 | 4 | 1 | 0.872 |
| T02_Box017_2017-10-05T11-29-03-818.jpg... | 13 | 16 | 11 | 5 | 2 | 0.759 |
| T02_Box017_2017-10-30T10-02-45-876.jpg... | 10 | 7 | 6 | 1 | 4 | 0.706 |
| T02_Box039_2017-09-07T11-14-04-723.jpg... | 18 | 0 | 0 | 0 | 18 | 0.000 |
| T02_Box039_2017-09-28T10-15-04-613.jpg... | 15 | 18 | 13 | 5 | 2 | 0.788 |
| T02_Box039_2017-10-12T10-40-25-533.jpg... | 5 | 4 | 4 | 0 | 1 | 0.889 |
| T02_Box039_2017-10-18T12-28-08-441.jpg... | 8 | 6 | 4 | 2 | 4 | 0.571 |
| T02_Box061_2017-09-07T10-47-24-525.jpg... | 9 | 1 | 0 | 1 | 9 | 0.000 |
| T02_Box061_2017-09-18T08-55-28-114.jpg... | 34 | 33 | 26 | 7 | 8 | 0.776 |
| T02_Box061_2017-09-26T14-54-24-598.jpg... | 17 | 17 | 12 | 5 | 5 | 0.706 |
| T02_Box061_2017-10-03T14-11-17-671.jpg... | 11 | 10 | 6 | 4 | 5 | 0.571 |
| T02_Box061_2017-10-16T08-42-40-150.jpg... | 6 | 8 | 6 | 2 | 0 | 0.857 |
| T02_Box061_2017-10-30T10-00-41-606.jpg... | 10 | 7 | 6 | 1 | 4 | 0.706 |

---
*Report generated by detection_evaluation.py*
