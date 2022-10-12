import hydra
from omegaconf import DictConfig, OmegaConf
from pipeline.pipeline import test_pipeline

@hydra.main(config_path=".conf", config_name="default.yaml")
def run_train(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    test_pipeline(config)


if __name__ == "__main__":
    run_train()
