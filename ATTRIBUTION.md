# Dataset attribution

VisionLoop uses a learning sample from **fashion-second-hand-front-only-rgb**,
published by **fnauman** on Hugging Face:
https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb

The dataset page lists **Creative Commons Attribution 4.0 International**:
https://creativecommons.org/licenses/by/4.0/

Changes made by this project: select five garment types; fetch viewer image
renditions; normalize orientation and RGB; resize/re-encode as JPEG; create local
train/validation/test assignments. Human-reviewed labels are recorded separately.
These changes do not imply endorsement by the dataset publisher.

The importer records the upstream train row number in each image's source field.
Source metadata should be retained if distributing a derived sample.
