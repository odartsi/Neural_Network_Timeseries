model_config = {
    "window_len": 7 * 3,  # Three weeks
    "forecast_len": 7,  # Seven days
    "aleatoric_features": [
        "daily_budget"
    ],  # List of features that cannot be used at forecast time
    "latent_dim": 16,
    "batch_size": 32,  # 16 for local test
    "epochs": 25,  # 2 for local test
    "n_units": [16, 16, 8],  # Dense layers number of hidden units
    "activations": ["relu", "relu", "relu"],  # Dense layers activation functions
    "optimiser": "adam",
    "learning_rate": 0.001,
    "loss": "huber",
    "metrics": ["mae", "mape"],  # Metrics list to be used to monitor the model
    "patience": 5,  # Number of epochs to wait before early stopping
    "path_to_save": "./models/checkpoints/",  # Path to save the model
    "pre-trained": False,  # Whether to start by a pre-trained model
    "pre-trained_path": "./models/pretrained/",  # Path to the pre-trained model
    "verbose": True,
}

preprocessing_config = {
    "GRANULARITY": [  # GRANULARITY constants for the grouping of the dataframe
        "tenant_id",
        "campaign_id",
        "channel",
        "section",
        "section_id",
        "app_id",
        "domain",
    ],
    "COLS_TO_KEEP": [
        "ad_spend",  # COLS_TO_KEEP constants
        "revenues",
        "channel",
        "app_id",
        "section",
        "domain",
        "daily_budget",
        "bid_amount",
        "year",
        "month",
        "day",
        "holiday",
        "weekday",
        "installs",
        "impressions",
        "clicks",
        "events",
        "transactions",
    ],
    "EXCLUDED_COLS": ["revenues", "ad_spend", "daily_budget", "bid_amount"],
    "EXCLUDED_DOMAINS": ["audible", "degpeg", "koober"],
    "N_NEIGHBORS": 4,
    "targets": ["ad_spend", "revenues"],
    "n_test_days": 28,
}
