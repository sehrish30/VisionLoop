# SageMaker reference — not validated on AWS

The local app never imports this folder. There is no boto3/SageMaker dependency,
AWS credential lookup, pipeline submission, training request, or deployment action.

`pipeline_reference.py` prints an **offline planning document**, including a
CreateTrainingJob argument shape and the local-to-cloud step mapping. It is not
an executable SageMaker Pipeline definition. `inference.py` contains reference
serving hooks sharing the local prediction transformation.

Still required before any real AWS use:

- Choose the tutorial's SageMaker SDK version and container image.
- Package the shared Python code with pinned compatible model dependencies.
- Implement processing/train/evaluation container entry points and S3 layout.
- Define real Pipeline SDK steps, their output dependencies, metrics property
  files, quality conditions, and model registration.
- Supply a role, bucket, region, and permissions; verify them on AWS separately.
- Implement endpoint deployment and retirement only when cloud usage is desired.

No endpoint code is run or installed by the local setup.

Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html
