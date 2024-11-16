# Submit a run using:
# python .\submit_mmlu_fewshot_knn_cot.py -cn knn_fewshot_cot_config

import time

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import omegaconf

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml import dsl, Input, MLClient
from azure.ai.ml.entities import Pipeline

from azureml_pipelines import (
    create_zeroshot_cot_pipeline,
    create_knn_fewshot_cot_pipeline,
)
from azureml_utils import get_component_collector
from configs import AMLConfig, KNNFewshotCoTPipelineConfig
from constants import GUIDANCE_PROGRAMS_DIR
from logging_utils import get_standard_logger_for_file

_logger = get_standard_logger_for_file(__file__)


@dataclass
class PipelineConfig:
    knn_fewshot_cot_config: KNNFewshotCoTPipelineConfig = omegaconf.MISSING
    azureml_config: AMLConfig = omegaconf.MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)


def create_mmlu_fewshot_knn_cot_pipeline(
    ml_client: MLClient, run_config: KNNFewshotCoTPipelineConfig, version_string: str
):
    components = get_component_collector(ml_client, version_string)

    zeroshot_cot_guidance_program = Input(
        type="uri_file",
        path=GUIDANCE_PROGRAMS_DIR / run_config.zeroshot_cot_guidance_program,
        model="download",
    )

    fewshot_cot_guidance_program = Input(
        type="uri_file",
        path=GUIDANCE_PROGRAMS_DIR / run_config.fewshot_cot_guidance_program,
        model="download",
    )

    answer_key = "multiple_choice_answer"
    cot_key = "chain_of_thought"

    @dsl.pipeline()
    def basic_pipeline() -> Pipeline:
        mmlu_fetch_job = components.jsonl_mmlu_fetch(
            mmlu_dataset=run_config.mmlu_dataset
        )
        mmlu_fetch_job.name = f"fetch_mmlu_{run_config.mmlu_dataset}"

        split_outputs = dict()
        for k, v in dict(
            input=run_config.test_split, example=run_config.example_split
        ).items():
            get_split_job = components.uri_folder_to_file(
                input_dataset=mmlu_fetch_job.outputs.output_dataset,
                filename_pattern=f"{v}.jsonl",
            )
            get_split_job.name = f"extract_split_{k}"
            split_outputs[k] = get_split_job.outputs.output_dataset

        # Now, get zero-shot CoTs for the examples
        example_cot_ds = create_zeroshot_cot_pipeline(
            pipeline_name=f"zeroshot_cot",
            pipeline_display_name=f"Zero Shot CoT",
            components=components,
            inference_config=run_config.aoai_config,
            input_dataset=split_outputs["example"],
            guidance_program=zeroshot_cot_guidance_program,
            output_key=answer_key,
            cot_key=cot_key,
        )

        filter_correct_job = components.jsonl_filter_correct_multiplechoice(
            input_dataset=example_cot_ds,
            response_key=answer_key,
            correct_key="correct_answer",
        )
        filter_correct_job.name = f"select_correct_responses"

        answer_ds = create_knn_fewshot_cot_pipeline(
            components=components,
            inference_config=run_config.aoai_config,
            embedding_config=run_config.aoai_embedding_config,
            input_dataset=split_outputs["input"],
            example_dataset=filter_correct_job.outputs.output_dataset,
            guidance_program=fewshot_cot_guidance_program,
            knn_config=run_config.knn_config,
            output_key=answer_key,
            cot_key=cot_key,
        )

        score_job = components.jsonl_score_multiplechoice(
            input_dataset=answer_ds,
            correct_key="correct_answer",  # Set when MMLU fetching
            response_key=answer_key,
        )
        score_job.name = f"fewshot_cot_score"

    pipeline = basic_pipeline()
    pipeline.experiment_name = (
        f"{run_config.pipeline.base_experiment_name}_{run_config.mmlu_dataset}"
    )
    pipeline.display_name = None
    pipeline.compute = run_config.pipeline.default_compute_target
    if run_config.pipeline.tags:
        pipeline.tags.update(run_config.tags)
    pipeline.tags.update(
        {
            "zeroshot_cot_program": run_config.zeroshot_cot_guidance_program,
            "fewshot_cot_program": run_config.fewshot_cot_guidance_program,
        },
    )
    _logger.info("Pipeline created")

    return pipeline


@hydra.main(config_path="configs", version_base="1.1")
def main(config: PipelineConfig):
    version_string = str(int(time.time()))
    _logger.info(f"AzureML object version for this run: {version_string}")

    _logger.info(f"Azure Subscription: {config.azureml_config.subscription_id}")
    _logger.info(f"Resource Group: {config.azureml_config.resource_group}")
    _logger.info(f"Workspace : {config.azureml_config.workspace_name}")

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    ws_client = MLClient(
        credential=credential,
        subscription_id=config.azureml_config.subscription_id,
        resource_group_name=config.azureml_config.resource_group,
        workspace_name=config.azureml_config.workspace_name,
        logging_enable=False,
    )

    pipeline = create_mmlu_fewshot_knn_cot_pipeline(
        ws_client, config.knn_fewshot_cot_config, version_string
    )
    _logger.info("Submitting pipeline")
    submitted_job = ws_client.jobs.create_or_update(pipeline)
    _logger.info(f"Submitted: {submitted_job.name}")


if __name__ == "__main__":
    main()
