# Data directory

Place the parallel corpus files here before running the experiment:

```text
data/
├── train.txt
└── test.txt
```

Each UTF-8 line must contain an English sentence and its Chinese translation separated by one tab:

```text
Hello.\t你好。
```

The original course split contained 26,918 training pairs and 2,991 test pairs. The release does not redistribute those files because their per-sentence attribution metadata was not included in the course package. The sentence content appears to originate from the Tatoeba/ManyThings English–Mandarin collection. If you prepare a new split from that collection, retain and publish the attribution fields required by the dataset license.

