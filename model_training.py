import random

import pandas as pd
import tensorflow as tf

import logging
from typing import Dict
from .utils.dataset_conversion import create_dataset
from .utils.train_tools import extract_number_features, get_callbacks, dense_block
from ...hooks.helpers.cli_mappings import CliKeyMap

random.seed(42)

logger = logging.getLogger(__name__)


def get_model(params: Dict, n_total_features: int) -> tf.keras.models.Model:
    """Function to build and compile the tensorflow model.

    Parameters:
    ----------
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.
    n_total_features : int
        number of features to build the input layer of the model.

    Returns
    -------
    tf.keras.models.Model
        compiled model instance.
    """
    logging.info("Building and compling Tensorflow model.")

    latent_dim = params["latent_dim"]  # Latent dimension of the model
    n_units, activations = (
        params["n_units"],
        params["activations"],
    )  # Numbers of hidden units and activation functions in the dense layers
    verbose = params["verbose"]

    # First branch of the net is a lstm which finds an embedding for the past
    past_inputs = tf.keras.Input(
        shape=(params["window_len"], n_total_features), name="past_inputs"
    )
    # Encoding the past
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True, name="Encoder")
    encoder_outputs, state_h, state_c = encoder(
        past_inputs
    )  # Apply the encoder object to past_inputs.
    n_deterministic_features = n_total_features - len(params["aleatoric_features"])
    future_inputs = tf.keras.Input(
        shape=(params["forecast_len"], n_deterministic_features), name="future_inputs"
    )
    # Combining future inputs with recurrent branch output
    decoder_lstm = tf.keras.layers.LSTM(
        latent_dim, return_sequences=True, name="Decoder"
    )
    x_revenues = decoder_lstm(future_inputs, initial_state=[state_h, state_c])
    x_adspend = x_revenues

    # Dense layers
    logger.debug(
        f"""Insert {len(n_units)} hidden Dense layers, with\n
        - units: {n_units}\n
        - activations: {activations}"""
    )

    x_adspend = dense_block(
        x_adspend, n_units=n_units, activations=activations, name_pattern="ad_spend"
    )
    x_revenues = dense_block(
        x_revenues, n_units=n_units, activations=activations, name_pattern="revenues"
    )
    # Output layers
    output_ad_spend = tf.keras.layers.Dense(1, name="ad_spend")(x_adspend)
    output_revenues = tf.keras.layers.Dense(1, name="revenues")(x_revenues)

    model = tf.keras.models.Model(
        inputs=[past_inputs, future_inputs], outputs=[output_ad_spend, output_revenues]
    )

    if params["optimiser"] == "adam":
        optimiser = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    else:
        logger.error(f"Invalid optimiser passed: {params['optimiser']}")
        raise NotImplementedError(f"Optimiser {params['optimiser']} not implemented")

    if isinstance(params["loss"], str):
        loss_params = tf.keras.losses.get(params["loss"])
        loss = {
            "ad_spend": loss_params,
            "revenues": loss_params,
        }
    else:
        loss_ads, loss_rev = tf.keras.losses.get(
            params["loss"][0]
        ), tf.keras.losses.get(params["loss"][1])
        loss = {
            "ad_spend": loss_ads,
            "revenues": loss_rev,
        }

    metrics = {"ad_spend": params["metrics"], "revenues": params["metrics"]}
    model.compile(loss=loss, optimizer=optimiser, metrics=metrics)

    if verbose:
        logger.info(f"Hyperparameters: \n{params}")
        logger.info(model.summary())

    return model


def train(
    model: tf.keras.models.Model,
    train_data: Dict[str, pd.DataFrame],
    params: Dict,
    run_params: Dict,
    pre_trained_model=None,
) -> tf.keras.models.Model:
    """Function to train the model.

    Parameters
    ----------
    model : tf.keras.models.Model
        model instance to be trained.
    train_data : Dict[str, pd.DataFrame]
        data to be used for training.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.
    run_params : Dict
        dictionary containing the relevant run parameters.

    Returns
    -------
    tf.keras.models.Model
        trained model instance.

    """
    logger.info("Training the Model.")
    verbose = params["verbose"]

    # Get model callbacks
    callbacks = get_callbacks(params)

    # Get the number of features
    n_total_features = extract_number_features(train_data)
    n_aleatoric_features = len(params["aleatoric_features"])
    n_deterministic_features = n_total_features - n_aleatoric_features

    logger.debug(
        f"""- n total features: {n_total_features}
            - n deterministic features: {n_deterministic_features}
            - n_aleatoric features: {n_aleatoric_features}

            - aleatoric_features: {params["aleatoric_features"]}"""
    )

    # Create dataset
    logger.debug("Convert dataframes to slices of tensorflow dataset")
    if params["validation_split"] is not None:
        train_dataset, val_dataset = create_dataset(
            train_data,
            n_deterministic_features=n_deterministic_features,
            window_size=params["window_len"],
            forecast_size=params["forecast_len"],
            batch_size=params["batch_size"],
            validation_split=params["validation_split"],
        )
    else:
        train_dataset = create_dataset(
            train_data,
            n_deterministic_features=n_deterministic_features,
            window_size=params["window_len"],
            forecast_size=params["forecast_len"],
            batch_size=params["batch_size"],
            validation_split=params["validation_split"],
        )

    is_backfill = run_params[f"{CliKeyMap.BACKFILL_KEY.value}"]
    try:
        if params["pre-trained"] or is_backfill:
            logger.debug(f"Using pre-trained: {params['pre-trained']}")
            logger.debug(
                f"Running backfill is {is_backfill} for {CliKeyMap.DATE_KEY.value}"
            )
            logger.info("Loading pre-trained weights.")
            logger.debug(
                f"""Loading pre-trained weights from {params["pre-trained_path"]}"""
            )
            pre_trained_model.save_weights(
                params["pre-trained_path"] + "pre_trained_weights.h5"
            )
            model.load_weights(params["pre-trained_path"] + "pre_trained_weights.h5")
    except Exception as e:
        logger.warning(
            f"Failed to load pre-trained weights for because of {e}. Calculating new weights.'"
        )

    # Train the model
    if params["validation_split"] is not None:
        model.fit(
            train_dataset,
            epochs=params["epochs"],
            callbacks=callbacks,
            verbose=verbose,
            validation_data=val_dataset,
        )
    else:
        model.fit(
            train_dataset,
            epochs=params["epochs"],
            callbacks=callbacks,
            verbose=verbose,
        )
    logger.debug("Model fitted.")
    return model


def get_model_and_train(
    train_data: Dict[str, pd.DataFrame],
    params: Dict,
    run_params: Dict,
    nn_pretrained_model: tf.keras.models.Model = None,
    acc: pd.DataFrame = None,
) -> tf.keras.models.Model:
    """Function to create a model with the right shape of inputs and then train it.

    Parameters
    ----------
    train_data : Dict[str, pd.DataFrame]
        data to be used for training.
    params : Dict
        dictionary containing the relevant hyperparameters and configurations.
    run_params : Dict
        dictionary containing the relevant run parameters.
    nn_pretrained_model: tf.keras.models.Model or None,
        Pre trained model containing info about model_wieghts.
    acc: pd.DataFrame or None
        DataFrame containing information about acc for ids.
    Returns
    -------
    tf.keras.models.Model
        trained model instance.

    """
    n_total_features = extract_number_features(train_data)
    logger.debug(f"Input shape: {(None, n_total_features)}")

    # get the accuracy file and check what is the percentage of predictions with accuracy greater or equal to 70
    if isinstance(acc, pd.DataFrame) and not acc.empty:
        compare_acc = True
    elif not acc:
        compare_acc = False
    if compare_acc:
        logger.info("Comparing accuracy of the pre-trained model.")
        df = (
            (acc["acc_roas"] >= params["accuracy_threshold"])
            .value_counts(normalize=True, sort=False)
            .reset_index(name="Percentage")
        )
        # if there are predictions with accuracy >=70% and represents at least the 10% of the predictions and
        # the average accuracy of all the predictions is greater than a threshold keep the model otherwise retrain the model

        if not df[df["index"] == True].empty:
            if (
                df.query("index==True")["Percentage"].iloc[0]
                >= params["predictions_percentage"]
                and acc.acc_roas.mean() >= params["mean_accuracy_limit"]
            ):
                logger.info("Loading pre-trained model.")
                # model_path = "data/06_models/neural_network/nn_trained.pkl"
                try:
                    if nn_pretrained_model or nn_pretrained_model["data"]:
                        return nn_pretrained_model
                except:
                    model = get_model(params, n_total_features)
                    logger.info("Model does not exist, start the training")
                    return train(
                        model, train_data, params, run_params, nn_pretrained_model
                    )

    model = get_model(params, n_total_features)
    keys=list(train_data.keys())
    print((train_data[keys[0]].columns))
    print(len(train_data[keys[0]].columns))
    logger.info("Starting the training")
    return train(model, train_data, params, run_params, nn_pretrained_model)
