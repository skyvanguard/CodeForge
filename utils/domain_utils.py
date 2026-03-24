DOMAIN_CONFIG = {
    "coding": {
        "score_key": "test_pass_rate",
        "splits": ["train"],
        "can_ensemble": False,
        "eval_subset": "",
        "test_subset": "",
        "stagedeval_samples": 10,
        "stagedeval_frac": 10 / 50,
        "has_val_subset": False,
    },
}


def get_domain_score_key(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["score_key"]


def get_domain_splits(domain, eval_test=False):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    splits = list(config["splits"])
    if eval_test and "test" not in splits:
        splits.append("test")
    return splits


def can_domain_ensembled(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["can_ensemble"]


def get_domain_eval_subset(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["eval_subset"]


def get_domain_test_subset(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["test_subset"]


def get_domain_stagedeval_samples(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["stagedeval_samples"]


def get_domain_stagedeval_frac(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["stagedeval_frac"]


def has_domain_val_subset(domain):
    config = DOMAIN_CONFIG.get(domain)
    if config is None:
        raise ValueError(f"Unknown domain: {domain}")
    return config["has_val_subset"]
