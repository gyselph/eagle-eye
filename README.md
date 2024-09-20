# EagleEye: Transformer-based malware detection using provenance graphs

This is the open source implementation of the [EagleEye paper](https://arxiv.org/abs/2408.09217), which was presented as a research paper at [eCrime 2024](https://apwg.org/event/ecrime2024/).

**Repository content**:

1. [EagleEye training](./1-EagleEye-training): Train a transformer from scratch to perform malware classification on sequence data.
2. *Coming soon:* [Malware dataset](./2-Malware-dataset): The malicious samples of the dataset "REE-2023", consisting of 7'000 provenance graphs.
3. *Coming soon:* [Feature extraction](3-Feature-extraction): The EagleEye data pipeline, which starts with raw behavior data and ends with behavior event sequences represented by rich security features.
4. *Coming soon:* [Security feature documentation](4-Security-features): A detailed description of all security features leveraged by EagleEye.
5. *Coming soon:* [ProvDetector re-implementation](5-ProvDetector): A re-implementation of one of the malware detection baselines.
6. *Coming soon:* [Command-line embedding](6-Command-line-embedding): The implementation for embedding command-line strings into high-dimensional vectors.
