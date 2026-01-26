# Local imports
import hashlib
import os

import hydra
import rootutils
import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from linear_cae.data import AudioDataset, TestAudioDataset
from linear_cae.inference import Autoencoder
from linear_cae.utils import RankedLogger, get_resolved_config_path_from_ckpt

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


def register_resolvers():
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("get_jamendo_paths", get_jamendo_paths)


def get_jamendo_paths(path: str, max) -> list[str]:
    """${[f'data/jamendo/audio/{i:02d}' for i in range(0, 99)]}"""
    log.info(f"Resolving Jamendo paths with max={max} at base path: {path}")
    return [os.path.join(path, f"jamendo/audio/{i:02d}") for i in range(0, max)]


register_resolvers()


def hash_config(cfg: DictConfig) -> str:
    """Hash the config to a string. This is used to create a unique name for the run."""
    hparams_str = str(cfg.seed)
    hparams_str += str(cfg.sample_rate)
    hparams_str += OmegaConf.to_yaml(cfg.trainer, resolve=True)
    hparams_str += OmegaConf.to_yaml(cfg.data, resolve=True)
    hparams_str += OmegaConf.to_yaml(cfg.model, resolve=True)

    return hashlib.md5(hparams_str.encode("utf-8")).hexdigest()[:8]


def resolve_and_dump_config(cfg: DictConfig) -> None:
    """Resolve and dump the config to a yaml file.

    Args:
        cfg: The configuration object
    """

    # Let's dump the resolved config so we can replicate the training easily without needing to
    # compose the config again
    config_data = OmegaConf.to_container(cfg, resolve=True)
    yaml_path = cfg.paths.output_dir + "/.hydra/config.yaml"
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path.replace("config", "config_resolved"), "w") as f:
        yaml.dump(config_data, f)

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config_path = cfg.paths.output_dir + "/.hydra/model.yaml"
    os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    return config_data


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """
    Main training launcher for Linear CAE.
    """
    seed_everything(cfg.get("seed", 42), workers=True, verbose=False)
    log.info(f"Setting seed to {cfg.get('seed', 42)}")
    log.info(f"Output dir: {cfg.paths.output_dir}")

    do_train = cfg.get("do_train", True)
    do_convert = cfg.get("do_convert", True)
    do_test = cfg.get("do_test", True)

    ckpt_path = cfg.get("ckpt_path", None)

    if do_train:
        cfg.paths.hash = hash_config(cfg)
        log.info(f"Instantiating data configs <{cfg.data}>")
        log.info(f"Instantiating dataset from {cfg.data.data_paths}")
        dataset = AudioDataset(
            cfg.data.data_paths,
            cfg.data.hop,
            cfg.data.fac,
            cfg.data.data_length,
            cfg.data.data_fractions,
            rms_min=cfg.data.rms_min,
            data_extensions=cfg.data.data_extensions,
            tot_samples=cfg.data.iters_per_epoch * cfg.data.batch_size,
            sample_rate=cfg.sample_rate,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data_loading.num_workers,
            prefetch_factor=(
                cfg.data_loading.prefetch_factor if cfg.data_loading.num_workers > 0 else None
            ),
            persistent_workers=(
                cfg.data_loading.persistent_workers if cfg.data_loading.num_workers > 0 else False
            ),
            drop_last=True,
            shuffle=True,
            pin_memory=cfg.data_loading.pin_memory,
        )

        config_data = resolve_and_dump_config(cfg)
    else:
        train_loader = None
        resolved_config_path = get_resolved_config_path_from_ckpt(ckpt_path)
        cfg = OmegaConf.load(resolved_config_path)
        config_data = OmegaConf.to_container(cfg, resolve=True)

    run_name = cfg.paths.hash

    if do_train or do_test:
        val_dataloaders = []
        validation_sets = cfg.data.get("validation_sets", None)
        if validation_sets is None:
            log.warning(f"No validation sets found in config for {cfg.data}. Is this intended?")
            validation_sets = {"jamendo_99": {"data_paths": cfg.data.data_path_test}}
            # ------ Ugly fix needed for legacy configs that don't have validation_sets
            OmegaConf.set_struct(cfg, False)
            cfg.data.validation_sets = validation_sets
            # Also update the callback's config so it gets the correct dict
            if "test_model" in cfg.callbacks:
                log.info("Updating test_model callback with validation sets...")
                cfg.callbacks.test_model.validation_sets = validation_sets
            OmegaConf.set_struct(cfg, True)

            # ------
        for name, val_set in validation_sets.items():
            val_dataset = TestAudioDataset(
                val_set["data_paths"],
                cfg.data.hop,
                cfg.data.fac,
                cfg.data.data_length_test,
                crop_to_length=cfg.data.crop_to_length,
                sample_rate=cfg.sample_rate,
                max_samples=cfg.data.get("max_val_samples", None),
            )

            val_dataloaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=cfg.data_loading.batch_size_val,
                    num_workers=cfg.data_loading.num_workers_val,
                    drop_last=False,
                    shuffle=False,
                    pin_memory=False,
                )
            )

    if do_test:
        if cfg.data.get("test_sets"):
            log.info("Instantiating test dataloaders...")
            test_dataloader = []
            for name, test_set_conf in cfg.data.test_sets.items():
                log.info(f"  - Loading test set: {name}")
                test_dataset = TestAudioDataset(
                    test_set_conf.data_paths,
                    cfg.data.hop,
                    cfg.data.fac,
                    cfg.data.data_length_test,
                    crop_to_length=cfg.data.crop_to_length,
                    sample_rate=cfg.sample_rate,
                    max_samples=cfg.data.get("max_val_samples", None),
                )
                test_dataloader.append(
                    DataLoader(
                        test_dataset,
                        batch_size=cfg.data_loading.batch_size_val,
                        num_workers=cfg.data_loading.num_workers_val,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=False,
                    )
                )
        else:
            log.warning("No test data found in config. Using validation data for testing.")
            test_dataloader = val_dataloaders

    # We rely on Hydra to instantiate the model with all flags from config
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # We initialize WandB manually for custom logging in training_step
    wandb_logger = WandbLogger(
        project=cfg.get("project", "linear_cae"),
        name=run_name,
        group=cfg.get("group", "default"),
        save_dir=cfg.paths.output_dir,
        log_model=False,
        id=run_name,
        config=config_data,
        notes=cfg.get("notes", ""),
    )
    # By running this we instantiate the wandb logger and create the run in wandb similar to wandb.init()
    wandb_logger.experiment

    # Use CSVLogger for the Trainer so it doesn't double-log to WandB
    logger = CSVLogger(
        save_dir=os.path.join(cfg.paths.output_dir, "csv"),
        name=None,
        version=None,
    )

    # Instantiate lightning callbacks
    callbacks: list[Callback] = []
    if cfg.get("callbacks", None) is not None:
        for _, cb_conf in cfg.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                # callbacks.append(hydra.utils.instantiate(cb_conf))
                cb_conf_resolved = OmegaConf.to_container(cb_conf, resolve=True)
                callbacks.append(hydra.utils.instantiate(cb_conf_resolved))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if do_train:
        log.info("Starting training...")
        trainer.fit(model, train_loader, val_dataloaders=val_dataloaders, ckpt_path=ckpt_path)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        do_convert = True  # If we finished training, we want to convert the checkpoint
        ckpt_type = "best"

    if do_test:
        if ckpt_path is not None:
            log.info(f"Testing model with checkpoint: {ckpt_path}")
        else:
            log.warning("No checkpoint found, using current model weights")

        model.strict_loading = True
        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)

    if do_convert:
        log.info("Converting checkpoint to inference format...")

        if ckpt_path:
            log.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=True)
            if "ema_state_dict" in ckpt:
                log.info("Loading EMA weights...")
                model.ema.load_state_dict(ckpt["ema_state_dict"])
                model.ema.copy_to()
            else:
                log.critical(
                    "No EMA weights found in checkpoint. Are you sure you trained with EMA?"
                )

        output_dir = os.path.join(cfg.paths.output_dir, "weights")
        os.makedirs(output_dir, exist_ok=True)

        # Save Weights
        torch.save(
            model.autoencoder_inference_model.state_dict(),
            os.path.join(output_dir, "autoencoder.pth"),
        )
        # Save Kwargs
        components = {
            "frontend": cfg.model.frontend,
            "generator": cfg.model.generator,
            "diffusion": cfg.model.diffusion,
        }

        for name, conf in components.items():
            components[name] = OmegaConf.to_container(conf, resolve=True)
            if "_target_" in components[name]:
                del components[name]["_target_"]
            if name == "frontend":
                components[name]["sample_rate"] = cfg.sample_rate

            with open(os.path.join(output_dir, f"{name}_kwargs.yaml"), "w") as f:
                yaml.dump(components[name], f)

        log.info(f"Model successfully converted. Artifacts saved to {output_dir}")

        # now we load from kwargs to test
        inference_model = Autoencoder(
            frontend=components["frontend"],
            generator=components["generator"],
            diffusion=components["diffusion"],
        )
        inference_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.paths.output_dir,
                    "weights",
                    "autoencoder.pth",
                ),
                map_location="cpu",
                weights_only=True,
            )
        )
        log.info(f"Successfully converted checkpoint: {cfg.ckpt_path}, type: {ckpt_type}")


if __name__ == "__main__":
    main()
