"""Offline SageMaker definition sketch. Standard library only; no AWS calls.

This writes a reference document, not a deployable pipeline. It intentionally
leaves resource configuration explicit and does not include an execution entry
point. Adapt these job arguments to the SDK version used by your tutorial.
"""
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CloudConfig:
    role_arn: str = "<SAGEMAKER_EXECUTION_ROLE_ARN>"
    image_uri: str = "<TRAINING_CONTAINER_IMAGE_URI>"
    dataset_uri: str = "s3://<BUCKET>/visionloop/dataset/"
    output_uri: str = "s3://<BUCKET>/visionloop/output/"
    instance_type: str = "ml.m5.large"


def training_job_arguments(config: CloudConfig):
    """Shape of CreateTrainingJob arguments, requiring an implemented container."""
    return {
        "RoleArn": config.role_arn,
        "AlgorithmSpecification": {"TrainingImage": config.image_uri, "TrainingInputMode": "File"},
        "InputDataConfig": [{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": config.dataset_uri, "S3DataDistributionType": "FullyReplicated"}},
        }],
        "OutputDataConfig": {"S3OutputPath": config.output_uri},
        "ResourceConfig": {"InstanceType": config.instance_type, "InstanceCount": 1, "VolumeSizeInGB": 10},
        "StoppingCondition": {"MaxRuntimeInSeconds": 1800},
    }


def reference_plan(config: CloudConfig):
    return {
        "status": "REFERENCE ONLY — not validated on AWS or deployable as-is",
        "configuration": asdict(config),
        "steps": [
            {"name": "PrepareDataset", "type": "Processing", "local_equivalent": "immutable snapshot + fixed splits"},
            {"name": "TrainCandidate", "type": "Training", "arguments": training_job_arguments(config)},
            {"name": "EvaluateBoth", "type": "Processing", "local_equivalent": "evaluate candidate and current model on identical validation data"},
            {"name": "QualityGate", "type": "Condition", "rule": "candidate macro F1 >= baseline macro F1 + 0.01; initial minimum 0.20"},
            {"name": "RegisterCandidate", "type": "RegisterModel", "approval": "PendingManualApproval"},
        ],
        "deployment": "Separate future step after approval; not implemented or invoked here.",
    }


if __name__ == "__main__":
    print(json.dumps(reference_plan(CloudConfig()), indent=2))
