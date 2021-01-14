Please organize this folder as follow:

```
./
├── config.py
├── ARID
│   ├── raw
│   │   ├── train_data -> (soft link to actual unlabeled training dataset)
│   │   │   ├── 0.mp4
│   │   │   ├── 1.mp4
│   │   │   ├── 2.mp4
│   │   │   ├── 3.mp4
│   │   │   ├── 4.mp4
│   │   │   ├── ...
│   │   ├── test_data -> (soft link to actual validation/testing dataset)
│   │   │   ├── 0.mp4
│   │   │   ├── 1.mp4
│   │   │   ├── 2.mp4
│   │   │   ├── 3.mp4
│   │   │   ├── 4.mp4
│   │   │   ├── ...
│   │   └── list_cvt
│   │       ├── ARID1.1_t2_train_pub.csv
│   │       ├── ARID1.1_t2_validation_gt_pub.csv
│   │       └── mapping_table_t2.txt
├── HMDB51
│   ├── raw
│   │   ├── train_data -> (soft link to actual labeled training dataset)
│   │   │   ├── drink (folder)
│   │   │   ├── jump
│   │   │   ├── pick
│   │   │   ├── pour
│   │   │   ├── push
│   │   └── list_cvt
│   │       ├── HMDB51_train_pub.csv
│   │       └── mapping_table_t2.txt
├── __init__.py
├── README.md
```