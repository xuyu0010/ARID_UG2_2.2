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
│   │   └── list_cvt
│   │       └── ug2_2022_train_dark.csv
├── Clear
│   ├── raw
│   │   ├── train_data -> (soft link to actual labeled training dataset)
│   │   │   ├── xxx.mp4
│   │   │   ├── xxx.mp4
│   │   │   ├── xxx.mp4
│   │   │   ├── xxx.mp4
│   │   │   ├── xxx.mp4
│   │   ├── test_data -> (soft link to actual dry-run dataset)
│   │   │   ├── 0.mp4
│   │   │   ├── 1.mp4
│   │   │   ├── 2.mp4
│   │   │   ├── 3.mp4
│   │   │   ├── 4.mp4
│   │   │   ├── ...
│   │   └── list_cvt
│   │       ├── hmdb_validate_public.csv
│   │       └── ug2_2022_train_labeled.csv
├── __init__.py
├── README.md
```