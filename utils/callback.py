from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN


def callbacks(rlr_monitor):
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8)
    ]
    return cbacks
