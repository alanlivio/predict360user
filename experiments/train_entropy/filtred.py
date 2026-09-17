import logging
import math
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf as oc
import predict360user as p3u

log = logging.getLogger()


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
        df = p3u.split_train_filtred(
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
        try:
            len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
            len_values = p3u.count_entropy(df[df["partition"] == "train"])
            log.info(f"Train samples: {dict(zip(len_keys, len_values))}")

            model = p3u.get_model(cfg)
            model.fit(df)
            eval_res = model.evaluate(df)
            return {"status": "success", "seed": cfg.seed, "eval": eval_res}
        except Exception as e:
            log.error(f"Error in train_and_eval_node for seed {cfg.seed}: {e}")
            raise e


def run(cfg: RunConfig, **kwargs) -> dict:
    assert cfg.train_entropy in p3u.ENTROPY_NAMES
    cfg.name = f"{cfg.model},filt={cfg.train_entropy}"
    if cfg.train_minsize:
        cfg.name += f",mins={cfg.train_minsize!r}"
    log.info("")
    log.info(f"==> run {cfg.name} with {cfg}")
    p3u.set_random_seed(cfg.seed)

    df = p3u.load_df_wins(
        dataset=cfg.dataset,
        init_window=cfg.init_window,
        h_window=cfg.h_window,
        m_window=cfg.m_window,
    )
    df = p3u.split_train_filtred(
        df,
        seed=cfg.seed,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        train_entropy=cfg.train_entropy,
        train_minsize=cfg.train_minsize,
    )

    len_keys = ["samples/train/all", "samples/train/low", "samples/train/medi", "samples/train/high"]
    len_values = p3u.count_entropy(df[df["partition"] == "train"])
    log.info(f"Train entropy: {dict(zip(len_keys, len_values))}")

    model = p3u.get_model(cfg)
    model.fit(df)

    eval_res = model.evaluate(df)
    del df
    del model
    return eval_res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = oc.from_cli()
    nseeds = args.pop("nseeds", 1)

    if RAY_AVAILABLE:
        log.info("Ray is available. Executing runs using Ray DAG...")
        ray.init(ignore_reinit_error=True)
        base_cfg = RunConfig(**args)
        df_node = load_and_split_node.bind(base_cfg)
        outputs = []
        for seed in range(0, nseeds):
            run_cfg = RunConfig(**args)
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
            cfg_run = RunConfig(**args)
            cfg_run.seed = seed
            run(cfg_run)
