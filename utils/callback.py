from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


def callbacks(early_stopping_patience):
    cbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8)
    ]
    return cbacks
