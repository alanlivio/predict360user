import logging
import math
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc
from sklearn.model_selection import train_test_split

import predict360user as p3u
import wandb

log = logging.getLogger()


def split_train_filtred(
    df: pd.DataFrame,
    train_entropy: str,
    seed: int = 1,
    train_size=p3u.RunConfig.train_size,
    test_size=p3u.RunConfig.test_size,
    val_size=0.25,
) -> pd.DataFrame:
    assert train_entropy in p3u.ENTROPY_NAMES
    df["partition"] = "discarted"  # sanity check
    log.info(f"{train_size=} (with {val_size=}), {test_size=}")

    # split train (with full size) and test
    train, test = train_test_split(
        df,
        random_state=seed,
        train_size=1 - test_size,
        test_size=test_size,
        stratify=df["actS_c"],
    )

    # sample train like was sample full len(df) to create similar size
    n_train = min(math.ceil(len(df) * train_size), len(train[train["actS_c"] == train_entropy]))
    train = train[train["actS_c"] == train_entropy].sample(n=n_train, random_state=seed)
    train, val = train_test_split(
        train,
        random_state=seed,
        test_size=val_size,
        stratify=train["actS_c"],
    )
    log.info("filtred samples/train are " + p3u.count_entropy_str(train))
    log.info("filtred samples/train/val are " + p3u.count_entropy_str(val))

    # save partition as column
    df.loc[train.index, "partition"] = "train"
    df.loc[val.index, "partition"] = "val"
    df.loc[test.index, "partition"] = "test"

    return df


@dataclass
class RunConfig(p3u.RunConfig):
    train_entropy: str = "all"
    train_minsize: bool = False
    num_cpus: int = 1
    num_gpus: int = 0


try:
    import ray
    from ray.dag import InputNode
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


if RAY_AVAILABLE:
    @ray.remote
    def load_and_split_node(cfg: RunConfig) -> pd.DataFrame:
        df = p3u.load_df_wins(
            dataset=cfg.dataset,
            init_window=cfg.init_window,
            h_window=cfg.h_window,
            m_window=cfg.m_window,
        )
        df = split_train_filtred(
            df,
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            train_entropy=cfg.train_entropy,
            seed=cfg.seed,
        )
        return df

    @ray.remote
    def train_and_eval_node(df: pd.DataFrame, cfg: RunConfig) -> dict:
        p3u.set_random_seed(cfg.seed)
        cfg.name = f"{cfg.model},filt={cfg.train_entropy}"
        if cfg.train_minsize:
            cfg.name += f",mins={cfg.train_minsize!r}"
        wandb.init(
            project=cfg.project,
            name=f"{cfg.name}-seed{cfg.seed}",
            group=cfg.name,
            job_type="train",
            reinit=True
        )
        try:
            len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
            len_values = p3u.count_entropy(df[df["partition"] == "train"])
            wandb.log(dict(zip(len_keys, len_values)), step=0)

            model = p3u.get_model(cfg)
            model.fit(df)
            model.evaluate(df)
            wandb.finish()
            return {"status": "success", "seed": cfg.seed}
        except Exception as e:
            log.error(f"Error in train_and_eval_node for seed {cfg.seed}: {e}")
            wandb.finish(exit_code=1)
            raise e


def run(cfg: RunConfig, **kwargs) -> None:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},filt={cfg.train_entropy}"
    if cfg.train_minsize:
        cfg.name += f",mins={cfg.train_minsize!r}"
    wandb.init(project=cfg.project, name=cfg.name, **kwargs)
    log.info("")
    log.info(f"==> run {cfg.name} with {cfg}")
    # set seed
    p3u.set_random_seed(cfg.seed)

    # load dataset
    df = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df = split_train_filtred(
        df,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        train_entropy=cfg.train_entropy,
        seed=cfg.seed,
    )

    # log train len
    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    wandb.log(dict(zip(len_keys, len_values)), step=0)

    # fit model
    model = p3u.get_model(cfg)
    model.fit(df)

    # evaluate model
    model.evaluate(df)

    # finish run
    wandb.finish()
    del df
    del model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = oc.from_cli()
    nseeds = args.pop('nseeds', 1)
    
    if RAY_AVAILABLE:
        log.info("Ray is available. Executing runs using Ray DAG...")
        ray.init(ignore_reinit_error=True)
        base_cfg = RunConfig(**args)  # type: ignore
        df_node = load_and_split_node.bind(base_cfg)
        outputs = []
        for seed in range(0, nseeds):
            run_cfg = RunConfig(**args)  # type: ignore
            run_cfg.seed = seed
            node = train_and_eval_node.options(
                num_cpus=run_cfg.num_cpus,
                num_gpus=run_cfg.num_gpus
            ).bind(df_node, run_cfg)
            outputs.append(node)
        
        @ray.remote
        def gather(*results):
            return list(results)
        
        dag = gather.bind(*outputs)
        results = ray.get(dag.execute())
        log.info(f"Ray DAG execution completed: {results}")
    else:
        log.info("Ray is not available. Falling back to sequential execution...")
        for seed in range(0, nseeds):
            CFG = RunConfig(**args)  # type: ignore
            CFG.seed = seed
            try:
                run(CFG)
            except:
                log.info(f"==> rerun falied {wandb.run.id} ")
                run(CFG, resume="must", id=wandb.run.id) # resume using same id

