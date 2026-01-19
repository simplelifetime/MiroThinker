# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

# Import from the new modular structure
from src.core.pipeline import (
    create_pipeline_components,
    execute_task_pipeline,
)
from src.logging.task_logger import bootstrap_logger

# Configure logger and get the configured instance
logger = bootstrap_logger()


async def amain(cfg: DictConfig) -> None:
    """Asynchronous main function."""

    logger.info(OmegaConf.to_yaml(cfg))

    # Create pipeline components using the factory function
    main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
        create_pipeline_components(cfg)
    )

    # Define task parameters
    task_id = "task_example"
    task_description = "Please answer the following question and also provide your problem-solving roadmap. Question: The image <image: 0> is a photo of a stadium. In a video titled containing \"Premium Experience\" published on the stadium's official YouTube channel in February 2016, there's a scene where fans are cheering for a hurling goal. What is the jersey number of the player who scored the goal?"
    task_file_name = ""
    # Optional: List of image URLs or local file paths for multimodal tasks
    # Examples:
    # image_urls = ["https://example.com/image.jpg"]  # URL
    # image_urls = ["/path/to/local/image.jpg"]       # Local file path
    image_urls = ["/home/liuzikang/MM-BrowseComp/MMBC_images/1.png"]

    # Execute task using the pipeline
    final_summary, final_boxed_answer, log_file_path, _ = await execute_task_pipeline(
        cfg=cfg,
        task_id=task_id,
        task_file_name=task_file_name,
        task_description=task_description,
        main_agent_tool_manager=main_agent_tool_manager,
        sub_agent_tool_managers=sub_agent_tool_managers,
        output_formatter=output_formatter,
        log_dir=cfg.debug_dir,
        image_urls=image_urls if image_urls else None,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(amain(cfg))


if __name__ == "__main__":
    main()
